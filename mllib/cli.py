import os
from dataclasses import fields

import torch
from fire import Fire

from .configs import (DataConfig, GlobalConfig, LoaderConfig, PathesConfig,
                      ROIFeaturesFRCNN, TrainConfig)
from .data import Data
from .extraction import Extract

PATH = os.path.dirname(os.path.realpath(__file__))


def clean_flags(flags, config):
    for field in fields(config):
        str_field = field.name
        if str_field in flags:
            flags.pop(str_field)


class Arguments(object):
    """ class to handle cli arguments"""

    def __init__(self, **kwargs):
        if not torch.cuda.is_available():
            kwargs["gpus"] = -1
        self.global_config = GlobalConfig(**kwargs)
        clean_flags(kwargs, self.global_config)
        self.dataset_config = DataConfig(**kwargs)
        clean_flags(kwargs, self.dataset_config)
        self.pathes_config = PathesConfig(**kwargs)
        clean_flags(kwargs, self.pathes_config)
        self.loader_config = LoaderConfig(**kwargs)
        clean_flags(kwargs, self.loader_config)
        self.flags = kwargs

    def train(self, name, **kwargs):
        pass

    def evaluate(self, name, **kwargs):
        pass

    def test(self, name, **kwargs):
        pass

    def pretrain(self, name, **kwargs):
        pass

    def download(self, name, **kwargs):
        pass

    def extract(self, input_dir, out_file):
        ids = self.flags.pop("ids", None)
        extract_config = ROIFeaturesFRCNN(out_file, input_dir, **self.flags)
        extractor = Extract(self.global_config, extract_config, ids=ids)
        print("initialized extractor, now starting run ...")
        extractor()

    def data(self, dataset, method):
        train_config = TrainConfig(**self.flags)
        data = Data(
            dataset_name=dataset,
            global_config=self.global_config,
            loader_config=self.loader_config,
            dataset_config=self.dataset_config,
            pathes_config=self.pathes_config,
            train_config=train_config,
        )
        assert hasattr(data, method)
        method_to_call = getattr(data, method)
        method_to_call()


def main():
    Fire(Arguments)
