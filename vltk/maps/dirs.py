import os

from datasets import Dataset
# other utils
from vltk.utils import get_classes

BASEPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEXTSETPATH = os.path.join(BASEPATH, "textset")
IMAGESETPATH = os.path.join(BASEPATH, "imageset")
EXPPATH = os.path.join(BASEPATH, "exp")
MODELPATH = os.path.join(BASEPATH, "modeling")


# class Models:
#     def __init__(self):
#         if "MODELS" not in globals():
#             global MODELS
#             MODELS = self._get_models()
#         if "MODELDICT" not in globals():
#             global MODELDICT
#             MODELDICT = {}

#     def _get_models(self):
#         source = (
#             open(MODELPATH + "/__init__.py").read().replace("\n", " ").split("import")
#         )
#         models = []
#         for s in source:
#             ms = s.replace(" ", "").split("from")[0].strip("(").strip(")")
#             for m in ms.split(","):
#                 if m:
#                     models.append(m)
#         return models

#     def add(self, model):
#         name = model.name if hasattr(model, "name") else model._name
#         MODELS[name] = model

#     def avail(self):
#         return list(MODELS.keys())

#     def get(self, model):
#         if model in MODELDICT:
#             return MODELDICT[model]
#         for s in MODELS:
#             if model.lower() == s.lower():
#                 return my_import(f"vltk.modeling.{s}")
#         raise Exception(f"could not find '{model}' out of avial: {self.avail_models()}")
