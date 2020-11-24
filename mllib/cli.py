from fire import Fire
import os
from .configs import Environment, ROIFeaturesFRCNN
from dataclasses import fields

PATH = os.path.dirname(os.path.realpath(__file__))

'''
this file is designed such that, only env variables defined in init
'''

class Arguments(object):
    """ class to handle cli arguments"""

    def __init__(self, **kwargs):
        self.environment_config = Environment()
        for field in fields(self.environment_config):
            str_field = field.name
            if str_field in kwargs:
                kwargs.pop(str_field)
        self.flags = kwargs

    def model(self, name, **kwargs):
        pass

    def extract(self, model, input_dir, out_file):
        self.extract_config = ROIFeaturesFRCNN(input_dir, out_file, **self.flags)
        '''
        init the model here
        once the model is started
        then pass the model and all the other things to the extract class
        '''
        print(self.extract_config, self.environment_config)


def main():
    Fire(Arguments)
