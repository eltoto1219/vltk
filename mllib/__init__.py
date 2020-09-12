from setuptools import setup

from .modeling_frcnn import GeneralizedRCNN
from .processing_image import Preprocess, img_tensorize
from .utils import Config, get_data, get_demo_path, get_image_from_url, load_checkpoint, load_labels
from .visualizing_image import SingleImageViz


"""
:)
"""
