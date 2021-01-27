import os

from setuptools import setup

IMAGEKEY = "img_id"
LABELKEY = "label"
TEXTKEY = "text"
SCOREKEY = "score"
RAWIMAGEKEY = "image"

# patche
BASEPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
TEXTSETPATH = os.path.join(BASEPATH, "textset")
IMAGESETPATH = os.path.join(BASEPATH, "imageset")
EXPPATH = os.path.join(BASEPATH, "exp")
SIMPLEPATH = os.path.join(BASEPATH, "simple")
MODELPATH = os.path.join(BASEPATH, "modeling")
LOOPPATH = os.path.join(BASEPATH, "loop")


"""
:)
"""
