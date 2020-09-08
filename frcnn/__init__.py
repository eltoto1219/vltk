from setuptools import setup

from .modeling_frcnn import GeneralizedRCNN
from .processing_image import Preprocess, tensorize
from .utils import Config, load_checkpoint, load_labels
from .visualizing_image import SingleImageViz


"""
:)
"""
