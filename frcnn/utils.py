"""
 coding=utf-8
 Copyright 2018, Antonio Mendoza Hao Tan, Mohit Bansal
 Adapted From Facebook Inc, Detectron2

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.import copy
 """

import copy
import json
import os
import pickle as pkl
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from yaml import Loader, dump, load

from .processing_image import Preprocess, tensorize
from .visualizing_image import SingleImageViz


PATH = "/".join(str(Path(__file__).resolve()).split("/")[:-1])
CONFIG = os.path.join(PATH, "config.yaml")
ATTRIBUTES = os.path.join(PATH, "attributes.txt")
OBJECTS = os.path.join(PATH, "objects.txt")
CHECKPOINT = os.path.join(PATH, "checkpoint.pkl")


def load_config(config=CONFIG):
    with open(config) as stream:
        data = load(stream, Loader=Loader)
    return Config(data)


def load_labels(objs=OBJECTS, attrs=ATTRIBUTES):
    vg_classes = []
    with open(objs) as f:
        for object in f.readlines():
            vg_classes.append(object.split(",")[0].lower().strip())

    vg_attrs = []
    with open(attrs) as f:
        for object in f.readlines():
            vg_attrs.append(object.split(",")[0].lower().strip())
    return vg_classes, vg_attrs


def load_checkpoint(ckp=CHECKPOINT):
    r = OrderedDict()
    with open(ckp, "rb") as f:
        ckp = pkl.load(f)["model"]
    for k in copy.deepcopy(list(ckp.keys())):
        v = ckp.pop(k)
        if isinstance(v, np.ndarray):
            v = torch.tensor(v)
        else:
            assert isinstance(v, torch.tensor), type(v)
        r[k] = v
    return r


class Config:
    _pointer = {}

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

        self._pointer = d

    def __repr__(self):
        return str(list((self._pointer.keys())))

    def __setattr__(self, key, val):
        self.__dict__[key] = val
        self.__dict__[key.upper()] = val
        levels = key.split(".")
        last_level = len(levels) - 1
        pointer = self._pointer
        if len(levels) > 1:
            for i, l in enumerate(levels):
                if hasattr(self, l) and isinstance(getattr(self, l), Config):
                    setattr(getattr(self, l), ".".join(levels[i:]), val)
                if l == last_level:
                    pointer[l] = val
                else:
                    pointer = pointer[l]

    def to_dict(self):
        return self._pointer

    def dump_yaml(self, data, file_name):
        with open(f"{file_name}", "w") as stream:
            dump(data, stream)

    def dump_json(self, data, file_name):
        with open(f"{file_name}", "w") as stream:
            json.dump(data, stream)

    def __str__(self):
        t = "    "
        if self._name != "root":
            r = f"{t * (self._level-1)}{self._name}:\n"
        else:
            r = ""
        level = self._level
        for i, (k, v) in enumerate(self._pointer.items()):
            if isinstance(v, Config):
                r += f"{t * (self._level)}{v}\n"
                self._level += 1
            else:
                r += f"{t * (self._level)}{k}: {v} ({type(v).__name__})\n"
            self._level = level
        return r[:-1]


if __name__ == "__main__":
    from .modeling_frcnn import GeneralizedRCNN

    im1 = "test_one.jpg"
    test_qustion = ["Is the man on a horse?"]
    target = "yes"
    # incase I want to batch
    img_tensors = list(map(lambda x: tensorize(x), [im1]))
    cfg = load_config()
    objids, attrids = load_labels()
    gqa_answers = json.load(open("gqa_answers.json"))
    # init classes
    visualizer = SingleImageViz(img_tensors[0], id2obj=objids, id2attr=attrids)
    preprocess = Preprocess(cfg)
    frcnn = GeneralizedRCNN(cfg)
    frcnn.load_state_dict(load_checkpoint(), strict=False)
    frcnn.eval()
    images, sizes, scales_yx = preprocess(img_tensors)
    output_dict = frcnn(images, sizes, scales_yx=scales_yx)
    # only want to select the first image
    output_dict = output_dict[0]
    features = output_dict.pop("roi_features")
    boxes = output_dict.pop("boxes")
    # add boxes and labels to the image
    visualizer.draw_boxes(
        boxes,
        output_dict.pop("obj_ids"),
        output_dict.pop("obj_scores"),
        output_dict.pop("attr_ids"),
        output_dict.pop("attr_scores"),
    )
    visualizer.save()
