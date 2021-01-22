from typing import List, Union

from vltk.abc.config import Config

__all__ = ["DeitConfig"]


class DeitConfig(Config):
    checkpoint = "/playpen1/home/avmendoz/data/deit/pytorch_model.bin"
    distillation_type = "soft"
    embed_dim = 576
    patch_size = 24
    img_size = 576
    mlp_ratio = 4.0
    num_heads = 12
    freeze_layers = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    qkv_bias = True
