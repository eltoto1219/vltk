from yaml import load, dump
from pprint import pprint

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def load_config():
    with open('config.yaml') as stream:
        data = load(stream, Loader=Loader)
    return data["config"]

def load_obj_data():
    vg_classes = []
    with open('objects_vocab.txt') as f:
        for object in f.readlines():
            vg_classes.append(object.split(',')[0].lower().strip())

    vg_attrs = []
    with open('attributes_vocab.txt') as f:
        for object in f.readlines():
            vg_attrs.append(object.split(',')[0].lower().strip())
    return vg_classes, vg_attrs

def load_ckp():
    with open("faster_rcnn_from_caffe_attr.pkl", "rb") as f:
        ckp = pkl.load(f)['model']
    return ckp


