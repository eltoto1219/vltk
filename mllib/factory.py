import os

import torch
from transformers import LxmertConfig, LxmertForQuestionAnswering

from mllib.compat import Config
from mllib.modeling.frcnn import FRCNN
from mllib.modeling.vit import VisionTransformer


def frcnn_factory(config):
    config = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    return FRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=config)


def lxmert_factory(config):
    ckp_name = getattr(config.models.lxmert, "ckp_transformers", None)
    lxmert_config = LxmertConfig(
        num_qa_labels=config.models.lxmert.known_labels,
        x_layers=config.models.lxmert.x_layers,
        l_layers=config.models.lxmert.l_layers,
        r_layers=config.models.lxmert.r_layers,
        task_matched=config.train.task_matched,
        task_mask_lm=config.train.task_mask_lm,
        task_obj_predict=config.train.task_obj_predict,
        visual_attr_loss=config.train.visual_attr_loss,
        visual_obj_loss=config.train.visual_obj_loss,
        visual_feat_loss=config.train.visual_feat_loss,
    )
    lxmert = LxmertForQuestionAnswering(lxmert_config)
    if ckp_name is not None:
        lxmert.from_pretrained(ckp_name)
    return lxmert


def vit_factory(config):
    pxls = config.data.img_max_size
    patch = config.data.patch_size
    vit_pretrained_dir = config.pathes.vit_pretrained_dir
    vitfp = os.path.join(vit_pretrained_dir, "pytorch_model.bin")
    vittorch = VisionTransformer(image_size=(pxls, pxls), patch_size=(patch, patch))
    vittorch.load_state_dict(torch.load(vitfp))
    for n, p in vittorch.named_parameters():
        if "embed" not in n:
            p.requires_grad = False
    return vittorch


NAME2MODEL = {
    "lxmert": lxmert_factory,
    "frcnn": frcnn_factory,
    "vit": vit_factory,
}


def model_name_to_instance(model_name, config):
    assert model_name is not None
    assert config is not None
    if model_name in NAME2MODEL:
        model = NAME2MODEL[model_name](
            config=config,
        )
        return model
    return None
