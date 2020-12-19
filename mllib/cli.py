import torch
from fire import Fire

from mllib import configs, loops, utils


class Main(object):
    """ class to handle cli arguments"""

    def __init__(self, **kwargs):
        if not torch.cuda.is_available():
            kwargs["gpu"] = -1
        kwargs = utils.load_yaml(kwargs)
        self.flags = kwargs

    def train(self, model, datasets):
        self.flags["model_name"] = model
        self.flags["datasets"] = datasets
        config = configs.GlobalConfig(**self.flags, command="train")
        aux_model = self.flags.get("aux_model", None)
        experiment = loops.Experiment(
            model=model, datasets=datasets, config=config, aux_model=aux_model
        )
        experiment()

    def eval(self, model, datasets):
        self.flags["model_name"] = model
        self.flags["datasets"] = datasets
        config = configs.GlobalConfig(**self.flags, command="eval")
        aux_model = self.flags.get("aux_model", None)
        experiment = loops.Experiment(
            model=model, datasets=datasets, config=config, aux_model=aux_model
        )
        experiment()

    def download(self, name, **kwargs):
        raise NotImplementedError()

    def extract(self, input_dir, out_file, preproc=None, name=None):
        raise NotImplementedError()

    def data(self, datasets, method=""):
        self.flags["datasets"] = datasets
        config = configs.GlobalConfig(**self.flags, command="data")
        data = loops.Data(datasets=datasets, config=config)
        if method == "":
            call = data
        else:
            call = getattr(data, method)
        call()


def main():
    Fire(Main)
