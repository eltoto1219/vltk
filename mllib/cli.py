import os
from dataclasses import fields

import torch
from fire import Fire
from tqdm import tqdm

from .configs import (DataConfig, GlobalConfig, LoaderConfig, PathesConfig,
                      ROIFeaturesFRCNN)
from .dataloader import BaseLoader
from .extracting_data import Extract

PATH = os.path.dirname(os.path.realpath(__file__))

"""
this file is designed such that, only env variables defined in init
"""


class Arguments(object):
    """ class to handle cli arguments"""

    def __init__(self, **kwargs):
        if not torch.cuda.is_available():
            kwargs["gpus"] = -1
        self.global_config = GlobalConfig(**kwargs)
        for field in fields(self.global_config):
            str_field = field.name
            if str_field in kwargs:
                kwargs.pop(str_field)
        self.flags = kwargs

    def model(self, name, **kwargs):
        pass

    def extract(self, model, input_dir, out_file):
        ids = self.flags.pop("ids", None)
        if ids is not None:
            ids = os.path.join(self.global_config.data_dir, ids)
            assert os.path.isfile(ids), f"no such id files {ids}"
        extract_config = ROIFeaturesFRCNN(out_file, input_dir, **self.flags)
        extractor = Extract(self.global_config, extract_config, ids=ids)
        print("initialized extractor, now starting run ...")
        extractor()

    def data(self, dataset, full_pass=False):
        dataset_config = DataConfig(**self.flags)
        pathes_config = PathesConfig(**self.flags)
        loader_config = LoaderConfig(**self.flags)
        global_config = self.global_config
        print("loaded all configs and flags")

        loader = BaseLoader(
            dataset, dataset_config, loader_config, global_config, pathes_config
        )
        print("intialized loader and dataset")

        for x in tqdm(loader):
            if not full_pass:
                for k, v in x.items():
                    print(k, v.shape)
                break


def main():
    Fire(Arguments)
