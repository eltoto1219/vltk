from transformers import (LxmertForPreTraining, LxmertForQuestionAnswering,
                          LxmertModel)

from .deit import DistilledVisionTransformer
from .frcnn import FRCNN
from .uniter import UniterModel
from .vit import VisionTransformer

__all__ = ["Get"]

Get = {
    "frcnn": FRCNN,
    "vit": VisionTransformer,
    "lxmert": LxmertModel,
    "lxmert_qa": LxmertForQuestionAnswering,
    "lxmert_pretraining": LxmertForPreTraining,
    "deit": DistilledVisionTransformer,
    "uniter": UniterModel,
}
