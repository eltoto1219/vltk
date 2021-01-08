import os

from datasets import Dataset
# other utils
from mllib.utils import IdentifierClass, get_classes, my_import

BASEPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEXTSETPATH = os.path.join(BASEPATH, "textset")
IMAGESETPATH = os.path.join(BASEPATH, "imageset")
EXPPATH = os.path.join(BASEPATH, "exp")
LOOPPATH = os.path.join(BASEPATH, "loop")
MODELPATH = os.path.join(BASEPATH, "modeling")


class Models:
    def __init__(self):
        if "MODELS" not in globals():
            global MODELS
            MODELS = self._get_models()
        if "MODELDICT" not in globals():
            global MODELDICT
            MODELDICT = {}

    def _get_models(self):
        source = (
            open(MODELPATH + "/__init__.py").read().replace("\n", " ").split("import")
        )
        models = []
        for s in source:
            ms = s.replace(" ", "").split("from")[0].strip("(").strip(")")
            for m in ms.split(","):
                if m:
                    models.append(m)
        return models

    def add(self, model):
        name = model.name if hasattr(model, "name") else model._name
        MODELS[name] = model

    def avail(self):
        return list(MODELS.keys())

    def get(self, model):
        if model in MODELDICT:
            return MODELDICT[model]
        for s in MODELS:
            if model.lower() == s.lower():
                return my_import(f"mllib.modeling.{s}")
        raise Exception(f"could not find '{model}' out of avial: {self.avail_models()}")


class Imagesets:
    def __init__(self):
        if "IMAGESETDICT" not in globals():
            global IMAGESETDICT
            IMAGESETDICT = get_classes(IMAGESETPATH, Dataset, pkg="mllib.imageset")

    def avail(self):
        return list(IMAGESETDICT.keys())

    def get(self, name):
        return IMAGESETDICT[name]

    def add(self, name, dset):
        IMAGESETDICT[name] = dset


class Textsets:
    def __init__(self):
        if "TEXTSETDICT" not in globals():
            global TEXTSETDICT
            TEXTSETDICT = get_classes(TEXTSETPATH, Dataset, pkg="mllib.textset")

    def avail(self):
        return list(TEXTSETDICT.keys())

    def get(self, name):
        return TEXTSETDICT[name]

    def add(self, name, dset):
        TEXTSETDICT[name] = dset


class Exps:
    def __init__(self):
        if "EXPDICT" not in globals():
            global EXPDICT
            EXPDICT = get_classes(EXPPATH, IdentifierClass, pkg="mllib.exp")

    def avail(self):
        return list(EXPDICT.keys())

    def get(self, name):
        return EXPDICT[name]

    def add(self, name, dset):
        EXPDICT[name] = dset


class Loops:
    def __init__(self):
        if "LOOPDICT" not in globals():
            global LOOPDICT
            LOOPDICT = get_classes(LOOPPATH, IdentifierClass, pkg="mllib.loop")

    def avail(self):
        return list(LOOPDICT.keys())

    def get(self, name):
        return LOOPDICT[name]

    def add(self, name, dset):
        LOOPDICT[name] = dset
