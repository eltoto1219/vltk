import copy
import json
import pickle as pkl
from collections import OrderedDict

import numpy as np
import PIL.Image as Image
import torch
from yaml import Loader, dump, load

from modeling_frcnn import GeneralizedRCNN
from processing_image import Preprocess, tensorize
from visualizing_image import SingleImageViz


def load_config(config="config.yaml"):
    with open(config) as stream:
        data = load(stream, Loader=Loader)
    return Config(data)


def load_obj_data(objs="objects.txt", attrs="attributes.txt"):
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
            v = torch.tensor(v)
        else:
            assert isinstance(v, torch.tensor), type(v)
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
    # load obj and attr labels
    objids, attrids = load_obj_data()
    # init image processor
    preprocess = Preprocess(cfg)
    # init model
    model = GeneralizedRCNN(cfg)
    # load the checkpoint
    model.load_state_dict(load_ckp(), strict=False)
    # prepare for inference
    model.eval()
    # tensorize the image
    img_tensor = tensorize(raw_img)
    # setup image viz
    visualizer = SingleImageViz(img_tensor, id2obj=objids, id2attr=attrids)
    # preprocess the image
    images, sizes, scale_yx = preprocess(img_tensor)
    # run model
    output_dict = model(images, sizes, scale_yx=scale_yx)
    # pop pooled features for later
    features = output_dict.pop("roi_features")
    # unsqueezing dictionary
    output_dict = dict(map(lambda i: (i[0], i[1].numpy()), output_dict.items()))
    # add boxes and labels to the image
    visualizer.draw_boxes(**output_dict)
    # save viz
    visualizer.save()
