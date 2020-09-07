from setuptools import setup

from inference_lxmert import Config, load_ckp, load_config, load_obj_data
from modeling_frcnn import GeneralizedRCNN
from processing_image import Preprocess, tensorize
from visualizing_image import SingleImageViz
