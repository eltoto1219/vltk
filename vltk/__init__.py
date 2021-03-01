import os

from setuptools import setup


def dummy():
    if False:
        return setup
    else:
        return


VLDATA = 0
VDATA = 1
LDATA = 2
ANNOTATION_DIR = "annotations"
IMAGEKEY = "img_id"
LABELKEY = "label"
TEXTKEY = "text"
SCOREKEY = "score"
RAWIMAGEKEY = "image"
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


# patche
BASEPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
TEXTSETPATH = os.path.join(BASEPATH, "textset")
IMAGESETPATH = os.path.join(BASEPATH, "imageset")
COMPLEXPATH = os.path.join(BASEPATH, "complex")
SIMPLEPATH = os.path.join(BASEPATH, "simple")
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
