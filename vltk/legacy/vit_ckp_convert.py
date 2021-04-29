# FOR CODE BORROWED FROM VISUALTRANSFORMER RELEASE TO CONVERT CHECKPOINTS:
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import collections
import os

import ml_collections
import numpy as np
import torch
from vltk.configs import Config, ModelConfig, PathesConfig, TrainConfig
from vltk.models.vit import VisionTransformer as VisionTransformerPytorch
from vltk. import convert_jax_to_torch_weights

# CONFIGS, KNOWN_MODELS,

# from vit_checkpoint


def _flatten_dict(d, parent_key="", sep="/"):
    """Flattens a dictionary, keeping empty leaves."""
    items = []
    for k, v in d.items():
        path = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(_flatten_dict(v, path, sep=sep).items())
        else:
            items.append((path, v))

    # Keeps the empty dict if it was set explicitly.
    if parent_key and not d:
        items.append((parent_key, {}))

    return dict(items)


def inspect_params(
    *, params, expected, logger, fail_if_extra=True, fail_if_missing=True
):
    """Inspects whether the params are consistent with the expected keys."""
    # params_flat = _flatten_dict(params)
    # expected_flat = _flatten_dict(expected)
    # missing_keys = expected_flat.keys() - params_flat.keys()
    # extra_keys = params_flat.keys() - expected_flat.keys()

    # Adds back empty dict explicitly, to support layers without weights.
    # Context: FLAX ignores empty dict during serialization.
    # empty_keys = set()
    # for k in missing_keys:
    #     if isinstance(expected_flat[k], dict) and not expected_flat[k]:
    #         params[k] = {}
    #         empty_keys.add(k)
    # missing_keys -= empty_keys

    # if empty_keys:
    # print("Inspect recovered empty keys:\n%s", empty_keys)
    # if missing_keys:
    #     print("Inspect missing keys:\n%s", missing_keys)
    # if extra_keys:
    #     print("Inspect extra keys:\n%s", extra_keys)
    #     raise ValueError(
    #         f"Missing params from checkpoint: {missing_keys}.\n"
    #         f"Extra params in checkpoint: {extra_keys}.\n"
    #         f"Restored params from checkpoint: {params_flat.keys()}.\n"
    #         f"Expected params from code: {expected_flat.keys()}."
    # )
    return params


def recover_tree(keys, values):
    """Recovers a tree as a nested dict from flat names and values.

    This function is useful to analyze checkpoints that are without need to access
    the exact source code of the experiment. In particular, it can be used to
    extract an reuse various subtrees of the scheckpoint, e.g. subtree of
    parameters.

    Args:
      keys: a list of keys, where '/' is used as separator between nodes.
      values: a list of leaf values.

    Returns:
      A nested tree-like dict.
    """
    tree = {}
    sub_trees = collections.defaultdict(list)
    for k, v in zip(keys, values):
        if "/" not in k:
            tree[k] = v
        else:
            k_left, k_right = k.split("/", 1)
            sub_trees[k_left].append((k_right, v))
    for k, kv_pairs in sub_trees.items():
        k_subtree, v_subtree = zip(*kv_pairs)
        tree[k] = recover_tree(k_subtree, v_subtree)
    return tree


def load(path):
    """Loads params from a checkpoint previously stored with `save()`."""
    with open(path, "rb") as f:
        ckpt_dict = np.load(f, allow_pickle=False)
        keys, values = zip(*list(ckpt_dict.items()))
    return recover_tree(keys, values)


def load_pretrained(*, pretrained_path, init_params, model_config, logger):
    """Loads/converts a pretrained checkpoint for fine tuning.

    Args:
      pretrained_path: File pointing to pretrained checkpoint.
      init_params: Parameters from model. Will be used for the head of the model
        and to verify that the model is compatible with the stored checkpoint.
      model_config: Configuration of the model. Will be used to configure the
        head and rescale the position embeddings.
      logger: Logger to use to output diagnostic messages.

    Returns:
      Parameters like `init_params`, but loaded with pretrained weights from
      `pretrained_path` and adapted accordingly.
    """

    restored_params = inspect_params(
        params=load(pretrained_path),
        expected=None,
        logger=logger,
        fail_if_extra=False,
        fail_if_missing=False,
    )

    # The following allows implementing fine-tuning head variants depending on the
    # value of `representation_size` in the fine-tuning job:
    # - `None` : drop the whole head and attach a nn.Linear.
    # - same number as in pre-training means : keep the head but reset the last
    #    layer (logits) for the new task.
    if model_config.representation_size is None:
        if "pre_logits" in restored_params:
            print("load_pretrained: drop-head variant")
            restored_params["pre_logits"] = {}
    # restored_params["head"]["kernel"] = init_params["head"]["kernel"]
    # restored_params["head"]["bias"] = init_params["head"]["bias"]

    if "posembed_input" in restored_params.get("Transformer", {}):
        pass
        # Rescale the grid of position embeddings. Param shape is (1,N,1024)
        # posemb = restored_params["Transformer"]["posembed_input"]["pos_embedding"]
        # posemb_new = init_params["Transformer"]["posembed_input"]["pos_embedding"]
        # if posemb.shape != posemb_new.shape:
        # pass
        # print(
        #     "load_pretrained: resized variant: %s to %s",
        #     posemb.shape,
        #     posemb_new.shape,
        # )
        # ntok_new = posemb_new.shape[1]

        # if model_config.classifier == "token":
        #     posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        #     ntok_new -= 1
        # else:
        #     posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

        # gs_old = int(np.sqrt(len(posemb_grid)))
        # gs_new = int(np.sqrt(ntok_new))
        # print("load_pretrained: grid-size from %s to %s", gs_old, gs_new)
        # posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

        # zoom = (gs_new / gs_old, gs_new / gs_old, 1)
        # posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
        # posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
        # posemb = jnp.array(np.concatenate([posemb_tok, posemb_grid], axis=1))
        # restored_params["Transformer"]["posembed_input"]["pos_embedding"] = posemb

    return restored_params


# my own code below:


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({"size": (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = "token"
    config.representation_size = None
    return config


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({"size": (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = "token"
    config.representation_size = None
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({"size": (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = "token"
    config.representation_size = None
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-H/14 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({"size": (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = "token"
    config.representation_size = None
    return config


MODEL_SIZES = {
    "ViT-B_16": 86_567_656,
    "ViT-B_32": 88_224_232,
    "ViT-L_16": 304_326_632,
    "ViT-L_32": 306_535_400,
    "ViT-H_14": 632_045_800,
    "testing": 2985,
}


def vit_jax_to_pytorch(model_config, train_config, config, pathes_config):
    vit_variant = model_config.vit_variant
    vit_pretrained_dir = pathes_config.vit_pretrained_dir
    vitfp = os.path.join(vit_pretrained_dir, vit_variant + ".npz")
    vittorchfp = os.path.join(vit_pretrained_dir, "pytorch_model.bin")
    os.makedirs(vit_pretrained_dir, exist_ok=True)

    # rng = np.zeros(2)
    # VisualTransformer = KNOWN_MODELS[vit_variant].partial(num_classes=1000)
    # output, initial_params = VisualTransformer.init_by_shape(
    #     rng, [((2, 832, 832, 3), np.float32)]
    # )
    # init_flat = flatten_dict(initial_params)
    params = load_pretrained(
        pretrained_path=vitfp,
        init_params=None,
        model_config=get_b32_config(),
        logger=None,
    )
    vittorch = VisionTransformerPytorch(image_size=(384, 384), patch_size=(16, 16))
    vitdict = vittorch.state_dict()
    torch_from_jax = convert_jax_to_torch_weights(torch_dict=vitdict, jax_dict=params)
    vittorch.load_state_dict(torch_from_jax)
    torch.save(vittorch.state_dict(), vittorchfp)


if __name__ == "__main__":
    # from google.colab import drive

    # drive.mount("/gdrive")
    # root = "/gdrive/My Drive/vision_transformer_colab"
    # import os

    # if not os.path.isdir(root):
    #     os.mkdir(root)
    #     os.chdir(root)
    # print(f'\nChanged CWD to "{root}"')
    model_config = ModelConfig
    train_config = TrainConfig
    config = Config
    pathes_config = PathesConfig
    vit_jax_to_pytorch(model_config, train_config, config, pathes_config)
