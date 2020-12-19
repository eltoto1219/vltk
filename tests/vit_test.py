import datetime
import os

import jax
import jax.numpy as jnp
import torch
# from sklearn.metrics import auc, roc_auc_score, roc_curve
from tqdm import tqdm
from transformers import AdamW  # LxmertForQuestionAnswering,

from mllib import BaseLoader, Evaluator
from mllib.configs import GlobalConfig, ModelConfig, PathesConfig, TrainConfig
from mllib.models import (CONFIGS, KNOWN_MODELS, VisionTransformerPytorch,
                          load_pretrained)
from mllib.utils import convert_jax_to_torch_weights, flatten_dict

MODEL_SIZES = {
    "ViT-B_16": 86_567_656,
    "ViT-B_32": 88_224_232,
    "ViT-L_16": 304_326_632,
    "ViT-L_32": 306_535_400,
    "ViT-H_14": 632_045_800,
    "testing": 2985,
}


def vit_jax_to_pytorch(model_config, train_config, global_config, pathes_config):
    vit_variant = model_config.vit_variant
    vit_pretrained_dir = os.path.join(
        global_config.data_dir, pathes_config.vit_pretrained_dir
    )
    vitfp = os.path.join(vit_pretrained_dir, vit_variant + ".npz")
    vittorchfp = os.path.join(vit_pretrained_dir, "pytorch_model.bin")
    os.makedirs(vit_pretrained_dir, exist_ok=True)

    rng = jax.random.PRNGKey(0)
    VisualTransformer = KNOWN_MODELS[vit_variant].partial(num_classes=1000)
    output, initial_params = VisualTransformer.init_by_shape(
        rng, [((2, 832, 832, 3), jnp.float32)]
    )
    # init_flat = flatten_dict(initial_params)
    params = load_pretrained(
        pretrained_path=vitfp,
        init_params=initial_params,
        model_config=CONFIGS[vit_variant],
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
    global_config = GlobalConfig
    pathes_config = PathesConfig
    vit_jax_to_pytorch(model_config, train_config, global_config, pathes_config)
