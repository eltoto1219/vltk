from transformers import LxmertConfig

from .deit import DeitConfig
from .frcnn import FRCNNConfig
from .vit import ViTConfig

__all__ = ["Get"]

Get = {
    "lxmert": LxmertConfig,
    "frcnn": FRCNNConfig,
    "vit": ViTConfig,
    "deit": DeitConfig,
}
