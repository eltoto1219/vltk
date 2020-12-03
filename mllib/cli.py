import os
from dataclasses import fields

import torch
from fire import Fire

from .configs import (DataConfig, GlobalConfig, LoaderConfig, ModelConfig,
                      PathesConfig, ROIFeaturesFRCNN, TrainConfig)
from .data import Data
from .extraction import Extract
from .mapping import NAME2MODEL
from .train import Trainer

PATH = os.path.dirname(os.path.realpath(__file__))


def clean_flags(flags, config):
    for field in fields(config):
        str_field = field.name
        if str_field in flags:
            flags.pop(str_field)


def model_name_to_instance(model_name, dataset_config, run_config, model_config):
    model = NAME2MODEL[model_name](
        command_name="train",
        model_config=model_config,
        run_config=run_config,
        dataset_config=dataset_config,
    )
    return model


class Arguments(object):
    """ class to handle cli arguments"""

    def __init__(self, **kwargs):
        if not torch.cuda.is_available():
            kwargs["gpus"] = -1
        self.global_config = GlobalConfig(**kwargs)
        self.dataset_config = DataConfig(**kwargs)
        self.pathes_config = PathesConfig(**kwargs)
        self.loader_config = LoaderConfig(**kwargs)
        self.flags = kwargs

    def train(self, model, dataset):
        train_config = TrainConfig(**self.flags)
        model_config = ModelConfig(**self.flags)
        model = model_name_to_instance(
            model_name=model,
            dataset_config=self.dataset_config,
            run_config=train_config,
            model_config=model_config,
        )
        trainer = Trainer(
            model=model,
            dataset_name=dataset,
            dataset_config=self.dataset_config,
            train_config=train_config,
            global_config=self.global_config,
            pathes_config=self.pathes_config,
            loader_config=self.loader_config,
            model_config=model_config,
        )
        trainer()

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
        # assert hasattr(data, method)
        method_to_call = getattr(data, method)
        method_to_call()


def main():
    Fire(Arguments)
