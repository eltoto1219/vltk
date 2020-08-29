import copy
import json
import pickle as pkl
from collections import OrderedDict

import numpy as np
import PIL.Image as Image
import torch
from yaml import Loader, dump, load

from image_processor_frcnn import PreProcess, img_array
from modeling_frcnn import GeneralizedRCNN


def load_config(config="config.yaml"):
    with open(config) as stream:
        data = load(stream, Loader=Loader)
    return Config(data)


def load_obj_data(objs="objects_vocab.txt", attrs="attributes_vocab.txt"):
    vg_classes = []
    with open(objs) as f:
        for object in f.readlines():
            vg_classes.append(object.split(",")[0].lower().strip())

    vg_attrs = []
    with open(attrs) as f:
        for object in f.readlines():
            vg_attrs.append(object.split(",")[0].lower().strip())
    return vg_classes, vg_attrs


def load_ckp(ckp="checkpoint.pkl"):
    r = OrderedDict()
    with open(ckp, "rb") as f:
        ckp = pkl.load(f)["model"]
    for k in copy.deepcopy(list(ckp.keys())):
        v = ckp.pop(k)
        if isinstance(v, np.ndarray):
            v = torch.Tensor(v)
        else:
            assert isinstance(v, torch.Tensor), type(v)
        r[k] = v
    return r


def show_image(a):
    a = np.uint8(np.clip(a, 0, 255))
    img = Image.fromarray(a)
    img.show()


def save_image(a, name="test_out", affix="jpg"):
    a = np.uint8(np.clipk(a, 0, 255))
    img = Image.fromarray(a)
    img.save(f"{name}.{affix}")


class Config:
    def __init__(self, dictionary: dict, name: str = "root", level=0):
        self._name = name
        self._level = level
        d = {}
        for k, v in dictionary.items():
            if v is None:
                raise ValueError()
            k = copy.deepcopy(k)
            v = copy.deepcopy(v)
            if isinstance(v, dict):
                v = Config(v, name=k, level=level + 1)
            d[k] = v
            setattr(self, k, v)
            setattr(self, k.upper(), getattr(self, k))

        self._pointer = d

    def __repr__(self):
        return str(list((self._pointer.keys())))

    def to_dict(self):
        return self._pointer

    def dump_yaml(self, data, file_name):
        with open(f"{file_name}", "w") as stream:
            dump(data, stream)

    def dump_json(self, data, file_name):
        with open(f"{file_name}", "w") as stream:
            json.dump(data, stream)

    def __str__(self):
        t = "  "
        r = f"{t * (self._level)}{self._name.upper()}:\n"
        level = self._level
        for i, (k, v) in enumerate(self._pointer.items()):
            if isinstance(v, Config):
                r += f"{t * (self._level)}{v}\n"
                self._level += 1
            else:
                r += f"{t * (self._level + 1)}{k}:{v}({type(v).__name__})\n"
            self._level = level
        return r[:-1]


if __name__ == "__main__":
    raw_img = "test_in.jpg"
    # load the config
    cfg = load_config()
    # init image processor
    preprocess = PreProcess(cfg)
    # init model
    model = GeneralizedRCNN(cfg)
    # load the checkpoint
    model.load_state_dict(load_ckp())
    # prepare for inference
    model.eval()
    # tensorize the image
    img = img_array(raw_img)
    # preprocess the image
    img, image_hw, scale_xy = preprocess(img)
    # run the backbone
    output_dict = model(
        img.unsqueeze(0), image_hw.unsqueeze(0), gt_boxes=None, proposals=None
    )
    boxes = output_dict["pred_boxes"]
    print(boxes, boxes.shape)
