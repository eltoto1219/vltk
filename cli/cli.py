from fire import Fire
import os

PATH = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])

class Arguments(object):
    """ class to handle cli arguments"""

    def __init__(self, _config=f"{PATH}/defaults.yaml", **kwargs):
        self.config = _config
        for k, v in kwargs.items():
            setattr(self, k, v)

    def model(self, name):
        print (f"here are the args you listed: {self.__dict__}")


def main():
    Fire(Arguments)
