import os

from setuptools import setup

# common keys across library
rawsize = "rawsize"
size = "size"
segmentation = "segmentation"
box = "box"
imgid = "imgid"
label = "label"
text = "text"
score = "score"
label = "label"
text = "text"
score = "score"
img = "image"
filepath = "filepath"
features = "features"
split = "split"
scale = "scale"
boxtensor = "boxtensor"
area = "area"
size = "size"


SPLITALIASES = {
    "testdev",
    "test",
    "dev",
    "eval",
    "val",
    "validation",
    "evaluation",
    "train",
}

# dataset selection values
VLDATA = 0
VDATA = 1
LDATA = 2

# pathes
ANNOTATION_DIR = "annotations"
BASEPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
VISIONLANG = os.path.join(BASEPATH, "adapters/visionlang")
VISION = os.path.join(BASEPATH, "adapters/vision")
EXTRACTION = os.path.join(BASEPATH, "adapters/vision")
ADAPTERS = os.path.join(BASEPATH, "adapters")
SIMPLEPATH = os.path.join(BASEPATH, "experiment")
MODELPATH = os.path.join(BASEPATH, "modeling")
LOOPPATH = os.path.join(BASEPATH, "loop")
SCHEDPATH = os.path.join(BASEPATH, "processing/sched.py")
DATAPATH = os.path.join(BASEPATH, "processing/data.py")
LABELPROCPATH = os.path.join(BASEPATH, "processing/label.py")
IMAGEPROCPATH = os.path.join(BASEPATH, "processing/image.py")
OPTIMPATH = os.path.join(BASEPATH, "processing/optim.py")


"""
:)
"""
