from setuptools import setup

from mllib import models

from .legacy_utils import (Config, get_data, get_demo_path, get_image_from_url,
                           load_checkpoint, load_labels)
from .processing_image import Preprocess, img_tensorize
from .visualizing_image import SingleImageViz

"""
:)
"""
