from setuptools import setup

from vltk.abc.extraction import VisnExtraction
from vltk.abc.visnadapter import VisnDataset
from vltk.abc.visnlangadatper import VisnLangDataset
from vltk.adapters import Adapters
from vltk.configs import DataConfig, LangConfig, VisionConfig
# from vltk.datasets import build
from vltk.dataset.builder import init_datasets
from vltk.features import Features
from vltk.processing import (LangProcessor, Processors, VisnLangProcessor,
                             VisnProcessor)
from vltk.vars import *


def build(config):
    return init_datasets(config)


"""
ALL BOXES ARE EXPECTED TO GO IN: (X, Y, W, H) FORMAT
"""


"""
:)
"""
