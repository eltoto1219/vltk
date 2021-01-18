from typing import Union, List

from vltk.abc.config import Config

__all__ = ["ViTConfig"]


class ViTConfig(Config):
    img_size: int = 800
    patches: int = 16
    pretrained_path: Union[
        None, str
    ] = "/playpen1/home/avmendoz/data/vit/pytorch_model.bin"
    freeze_embeddings: bool = False
    # should match number of layers present in the model
    freeze_layers: List[bool] = [1] * 12
    freeze_heads: bool = True
    dropout: float = 0.1
