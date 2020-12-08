import datetime
import os

import jax
import jax.numpy as jnp
import torch
# from sklearn.metrics import auc, roc_auc_score, roc_curve
from tqdm import tqdm
from transformers import AdamW  # LxmertForQuestionAnswering,

from mllib import BaseLoader, Evaluator, flatten_dict
from mllib.configs import GlobalConfig, ModelConfig, PathesConfig, TrainConfig
from mllib.models import (CONFIGS, KNOWN_MODELS, VisionTransformerPytorch,
                          load_pretrained)

MODEL_SIZES = {
    "ViT-B_16": 86_567_656,
    "ViT-B_32": 88_224_232,
    "ViT-L_16": 304_326_632,
    "ViT-L_32": 306_535_400,
    "ViT-H_14": 632_045_800,
    "testing": 2985,
}


def init_vit(model_config, train_config, global_config, pathes_config):
    vit_variant = model_config.vit_variant
    vit_pretrained_dir = os.path.join(
        global_config.data_dir, pathes_config.vit_pretrained_dir
    )
    vitfp = os.path.join(vit_pretrained_dir, vit_variant + ".bin")
    os.makedirs(vit_pretrained_dir, exist_ok=True)

    # return vit
    rng = jax.random.PRNGKey(0)
    VisualTransformer = KNOWN_MODELS[vit_variant].partial(num_classes=0)
    output, initial_params = VisualTransformer.init_by_shape(
        rng, [((2, 832, 832, 3), jnp.float32)]
    )
    flattened_params = flatten_dict(
        initial_params["Transformer"], parent_key="transformer"
    )
    vittorch = VisionTransformerPytorch()
    vitdict = vittorch.state_dict()
    for (jk, jv), (tk, tv) in zip(vitdict.items(), flattened_params.items()):
        print(jk, jv.shape, "|", tk, tv.shape)
    # print(set(flattened_params) - set(vitdict))


if __name__ == "__main__":
    model_config = ModelConfig
    train_config = TrainConfig
    global_config = GlobalConfig
    pathes_config = PathesConfig
    init_vit(model_config, train_config, global_config, pathes_config)
