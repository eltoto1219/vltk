from setuptools import setup

from mllib import models

from .legacy_processing import Preprocess, img_tensorize
from .transformers_compat import (Config, get_data, get_demo_path,
                                  get_image_from_url, load_checkpoint,
                                  load_labels)
from .visualize import SingleImageViz

"""
:)
"""
