import math
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import gelu, swish
from transformers.configuration_utils import PretrainedConfig
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import (PreTrainedModel,
                                         apply_chunking_to_forward,
                                         find_pruneable_heads_and_indices,
                                         prune_linear_layer)
from transformers.tokenization_bert import BertTokenizer, BertTokenizerFast

# SECTION 1: tokenization

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "visualbert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    }
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"visualbert-base-uncased": 512}
PRETRAINED_INIT_CONFIGURATION = {"visualbert-base-uncased": {"do_lower_case": True}}


class VisualBertTokenizer(BertTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION


class VisualBertTokenizerFast(BertTokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION


# SECTION 2: modeling

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}
VisualBertLayerNorm = torch.nn.LayerNorm


class VisualBertConfig(PretrainedConfig):

    model_type = "visualbert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        num_labels=2,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        visual_feat_dim=2048,
        visual_pos_dim=4,
        output_hidden_states=False,
        output_attentions=False,
        bypass_transformer=False,
        add_boxes=False,
        pooler_strategy="default",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_labels = num_labels
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.visual_feat_dim = visual_feat_dim
        self.visual_pos_dim = visual_pos_dim
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.num_hidden_layers = num_hidden_layers
        self.bypass_transformer = bypass_transformer
        self.add_boxes = add_boxes
        self.pooler_strategy = pooler_strategy


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


@dataclass
class VisualBertModelOutput(ModelOutput):
    last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class VisualBertForQuestionAnsweringOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    question_answering_score: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    pooled_output: Optional[Tuple[torch.FloatTensor]] = None


def load_tf_weights_in_visualbert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    import re

    import numpy as np
    import tensorflow as tf

    tf_path = os.path.abspath(tf_checkpoint_path)
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n
            in [
                "adam_v",
                "adam_m",
                "AdamWeightDecayOptimizer",
                "AdamWeightDecayOptimizer_1",
                "global_step",
            ]
            for n in name
        ):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        pointer.data = torch.from_numpy(array)
    return model


class VisualBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in VisualBertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs


class VisualBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = VisualBertLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class VisualBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = VisualBertSelfAttention(config)
        self.output = VisualBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class VisualBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class VisualBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = VisualBertLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class VisualBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = VisualBertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert (
                self.is_decoder
            ), f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = VisualBertAttention(config)
        self.intermediate = VisualBertIntermediate(config)
        self.output = VisualBertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:]
            )  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class VisualBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [VisualBertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_attentions]
                if v is not None
            )
        return VisualBertModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class VisualBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class VisualBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=0
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size, padding_idx=0
        )

        self.LayerNorm = VisualBertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        seq_length = input_shape[1]

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class VisualBertAnswerHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        hid_dim = config.hidden_size
        self.ll1 = nn.Linear(hid_dim, hid_dim)
        self.LayerNorm = VisualBertLayerNorm(hid_dim, eps=1e-12)
        self.ll2 = nn.Linear(hid_dim, num_labels)

    def forward(self, hidden_states):
        return self.ll2(self.LayerNorm(gelu(self.ll1(hidden_states))))


class VisualBertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = VisualBertConfig
    load_tf_weights = load_tf_weights_in_visualbert
    base_model_prefix = "visualbert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, VisualBertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class VisualBertMultiModalEmbeddings(VisualBertEmbeddings):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

        # apperently, there is no difference between this and the other thing.
        self.token_type_embeddings_visual = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.position_embeddings_visual = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.projection = nn.Linear(config.visual_feat_dim, config.hidden_size)
        if config.add_boxes:
            self.box_projection = nn.Linear(config.visual_pos_dim, config.hidden_size)

    def initialize_visual_from_pretrained(self):
        self.token_type_embeddings_visual.weight = nn.Parameter(
            deepcopy(self.token_type_embeddings.weight.data), requires_grad=True
        )
        self.position_embeddings_visual.weight = nn.Parameter(
            deepcopy(self.position_embeddings.weight.data), requires_grad=True
        )

    def encode_text(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        return embeddings

    def encode_image(
        self,
        visual_feats,
        visual_feature_type_ids,
        image_text_alignment,
        visual_pos=None,
    ):

        visual_feats = self.projection(visual_feats)
        if self.config.add_boxes and visual_pos is not None:
            ve = self.visn_layer_norm(visual_feats)
            be = self.box_fc(visual_pos)
            be = self.box_layer_norm(be)
            visual_feats = (ve + be) / 2

        # token_type_embeddings_visual = self.token_type_embeddings_visual(visual_feature_type_ids)
        token_type_embeddings_visual = self.token_type_embeddings(
            visual_feature_type_ids
        )

        # get position_embeddings
        # this depends on image_text_alignment
        position_embeddings_visual = self.get_position_embeddings_visual(
            visual_feats, image_text_alignment=image_text_alignment
        )

        # calculate visual embeddings
        v_embeddings = (
            visual_feats + position_embeddings_visual + token_type_embeddings_visual
        )
        return v_embeddings

    def get_position_embeddings_visual(self, visual_feats, image_text_alignment):

        if image_text_alignment is not None:
            # image_text_alignment = Batch x number of ROI features x alignment_number.
            # Each element denotes the position of the word corresponding to the
            image_text_alignment_mask = (
                (image_text_alignment != self.config.pad_token_id)
                .long()
                .to(image_text_alignment.device)
            )
            # Get rid of the pad.
            image_text_alignment = image_text_alignment_mask * image_text_alignment

            # = Batch x image_length x alignment length x dim
            position_embeddings_visual = self.position_embeddings(
                image_text_alignment
            ) * image_text_alignment_mask.unsqueeze(-1)
            position_embeddings_visual = position_embeddings_visual.sum(2)

            # We want to averge along the alignment_number dimension.
            image_text_alignment_mask = image_text_alignment_mask.sum(2)
            image_text_alignment_mask[image_text_alignment_mask == 0] = torch.tensor(
                [1], dtype=torch.long
            )  # Avoid devide by zero error
            position_embeddings_visual = (
                position_embeddings_visual / image_text_alignment_mask.unsqueeze(-1)
            )

            position_ids_visual = torch.zeros(
                visual_feats.size()[:-1],
                dtype=torch.long,
                device=visual_feats.device,
            )

            position_embeddings_visual = (
                position_embeddings_visual
                + self.position_embeddings_visual(position_ids_visual)
            )
        else:
            position_ids_visual = torch.zeros(
                visual_feats.size()[:-1],
                dtype=torch.long,
                device=visual_feats.device,
            )
            position_embeddings_visual = self.position_embeddings_visual(
                position_ids_visual
            )

        return position_embeddings_visual

    def forward(
        self,
        input_ids,
        token_type_ids,
        visual_feats,
        visual_feature_type_ids,
        image_text_alignment,
        visual_pos=None,
    ):

        # text embeddings
        text_embeddings = self.encode_text(input_ids, token_type_ids=token_type_ids)

        # visual embeddings
        if visual_feats is not None and visual_feature_type_ids is not None:
            v_embeddings = self.encode_image(
                visual_feats,
                visual_feature_type_ids=visual_feature_type_ids,
                image_text_alignment=image_text_alignment,
                visual_pos=visual_pos,
            )

            # Concate the two:
            embeddings = torch.cat(
                (text_embeddings, v_embeddings), dim=1
            )  # concat the visual embeddings after the attentions

        else:
            embeddings = text_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class VisualBertModel(VisualBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = VisualBertMultiModalEmbeddings(config)
        self.encoder = VisualBertEncoder(config)
        self.pooler = VisualBertPooler(config)
        self.bypass_transformer = config.bypass_transformer

        if self.bypass_transformer:
            self.additional_layer = VisualBertLayer(config)

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        visual_feats,
        visual_feature_type_ids=None,
        image_text_alignment=None,
        visual_pos=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids,
            token_type_ids,
            visual_feats=visual_feats,
            visual_feature_type_ids=visual_feature_type_ids,
            image_text_alignment=image_text_alignment,
            visual_pos=visual_pos,
        )

        if self.bypass_transformer and visual_feats is not None:
            text_length = input_ids.size(1)
            text_embedding_output = embedding_output[:, :text_length, :]
            visual_part = embedding_output[:, text_length:, :]

            text_extended_attention_mask = extended_attention_mask[
                :, :, :text_length, :text_length
            ]

            encoded_layers = self.encoder(
                text_embedding_output, text_extended_attention_mask
            )
            sequence_output = encoded_layers[0]
            new_input = torch.cat((sequence_output, visual_part), dim=1)
            final_sequence_output = self.additional_layer(
                new_input, extended_attention_mask
            )
            pooled_output = self.pooler(final_sequence_output[0])
            return final_sequence_output[0], pooled_output, []
        else:
            # if only textual data is passed
            encoded_layers = self.encoder(embedding_output, extended_attention_mask)
            sequence_output = encoded_layers[0]
            pooled_output = self.pooler(sequence_output)
            attn_data_list = []

            if self.output_attentions:
                attn_data_list = encoded_layers[1:]

            return sequence_output, pooled_output, attn_data_list


class VisualBertForQuestionAnswering(VisualBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.pooler_strategy = self.config.pooler_strategy
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.visualbert = VisualBertModel(config)
        self.init_weights()
        self.classifier = VisualBertAnswerHead(config, self.num_labels)

        # Weight initialization

        # Loss function
        self.aux_classifier = VisualBertAnswerHead(config, 10)
        self.loss = CrossEntropyLoss()
        self.extra_loss = CrossEntropyLoss()

    def resize_num_qa_labels(self, num_labels):
        cur_qa_logit_layer = self.get_qa_logit_layer()
        if num_labels is None or cur_qa_logit_layer is None:
            return
        new_qa_logit_layer = self._resize_qa_labels(num_labels)
        self.config.num_labels = num_labels
        self.num_labels = num_labels

        return new_qa_logit_layer

    def _resize_qa_labels(self, num_labels):
        cur_qa_logit_layer = self.get_qa_logit_layer()
        new_qa_logit_layer = self._get_resized_qa_labels(cur_qa_logit_layer, num_labels)
        self._set_qa_logit_layer(new_qa_logit_layer)
        return self.get_qa_logit_layer()

    def get_qa_logit_layer(self) -> nn.Module:
        if hasattr(self, "classifier"):
            return self.classifier.logit_fc[-1]

    def _set_qa_logit_layer(self, qa_logit_layer):
        self.classifier.logit_fc[-1] = qa_logit_layer

    def _get_resized_qa_labels(self, cur_qa_logit_layer, num_labels):

        if num_labels is None:
            return cur_qa_logit_layer

        cur_qa_labels, hidden_dim = cur_qa_logit_layer.weight.size()
        if cur_qa_labels == num_labels:
            return cur_qa_logit_layer

        # Build new linear output
        if getattr(cur_qa_logit_layer, "bias", None) is not None:
            new_qa_logit_layer = nn.Linear(hidden_dim, num_labels)
        else:
            new_qa_logit_layer = nn.Linear(hidden_dim, num_labels, bias=False)

        new_qa_logit_layer.to(cur_qa_logit_layer.weight.device)

        # initialize all new labels
        self.init_weights(new_qa_logit_layer)

        # Copy labels from the previous weights
        num_labels_to_copy = min(cur_qa_labels, num_labels)
        new_qa_logit_layer.weight.data[
            :num_labels_to_copy, :
        ] = cur_qa_logit_layer.weight.data[:num_labels_to_copy, :]
        if getattr(cur_qa_logit_layer, "bias", None) is not None:
            new_qa_logit_layer.bias.data[
                :num_labels_to_copy
            ] = cur_qa_logit_layer.bias.data[:num_labels_to_copy]

        return new_qa_logit_layer

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        visual_feats,
        visual_feature_type_ids=None,
        image_text_alignment=None,
        visual_pos=None,
        labels=None,
        return_dict=False,
        extras=None,
    ):
        sequence_output, pooled_output, attention_weights = self.visualbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            visual_feats=visual_feats,
            visual_feature_type_ids=visual_feature_type_ids,
            image_text_alignment=image_text_alignment,
            visual_pos=visual_pos,
        )

        if self.pooler_strategy == "vqa":
            index_to_gather = attention_mask.sum(1) - 2
            pooled_output = torch.gather(
                sequence_output,
                1,
                index_to_gather.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(index_to_gather.size(0), 1, sequence_output.size(-1)),
            )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = logits.contiguous().view(-1, self.num_labels)
        loss = None
        if labels is not None:
            loss = self.loss(logits, labels.view(-1))
        if extras is not None:
            aux_logits = self.aux_classifier(pooled_output)
            loss += self.loss(aux_logits, extras.view(-1))

        if not return_dict:
            return VisualBertForQuestionAnsweringOutput(
                loss=loss,
                question_answering_score=logits,
                hidden_states=sequence_output,
                attentions=attention_weights,
                pooled_output=pooled_output,
            )
        else:
            return {
                "loss": loss,
                "question_answering_score": logits,
                "hidden_states": sequence_output,
                "attentions": attention_weights,
                "pooled_output": pooled_output,
            }
