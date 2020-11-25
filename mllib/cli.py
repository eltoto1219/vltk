import torch
from fire import Fire
import os
from .configs import Environment, ROIFeaturesFRCNN
from dataclasses import fields
from .extracting_data import Extract

PATH = os.path.dirname(os.path.realpath(__file__))

'''
this file is designed such that, only env variables defined in init
'''

class Arguments(object):
    """ class to handle cli arguments"""

    def __init__(self, **kwargs):
        if not torch.cuda.is_available():
            kwargs["gpus"] = -1
        self.environment_config = Environment(**kwargs)
        for field in fields(self.environment_config):
            str_field = field.name
            if str_field in kwargs:
                kwargs.pop(str_field)
        self.flags = kwargs

    def model(self, name, **kwargs):
        pass

    def extract(self, model, input_dir, out_file):
        self.extract_config = ROIFeaturesFRCNN(out_file, input_dir, **self.flags)
        extractor = Extract(self.environment_config, self.extract_config)
        print("initialized extractor, now starting run ...")
        extractor()


def main():
    Fire(Arguments)
