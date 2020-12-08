from setuptools import setup

from mllib import models

from .configs import *
from .dataloader import BaseLoader
from .evaluate import Evaluator
from .legacy_processing import Preprocess, img_tensorize
from .transformers_compat import (Config, get_data, get_demo_path,
                                  get_image_from_url, load_checkpoint,
                                  load_labels)
from .utils import flatten_dict
from .visualize import SingleImageViz

"""
:)
"""
