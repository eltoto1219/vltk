import atexit
import os
import sys
from io import StringIO

import torch
from fire import Fire

from mllib import configs, utils
from mllib.abc.experiment import Experiment

STDERR = sys.stderr = StringIO()
EXPPATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")), "exp")


@atexit.register
def crash_save():
    errorlog = STDERR.getvalue()
    if "experiment" in globals():
        if experiment is not None and config is not None:
            save_on_crash = getattr(config, "save_on_crash", False)
            if config.email is not None:
                utils.send_email(config.email, errorlog)
            if save_on_crash:
                try:
                    experiment.save()
                    print("\nCRASH SAVE SUCCESS: mllib command crashed and was saved")
                except Exception:
                    print("\nFAILURE: mllib command crashed and was not saved")
            else:
                print("\nWARNING: mllib command crashed and was not saved")


@atexit.register
def restore_stdout():
    sys.stderr = sys.__stderr__
    sys.stderr.write(STDERR.getvalue())
    print()


class Main(object):
    """ class to handle cli arguments"""

    def __init__(self, **kwargs):
        print()
        if not torch.cuda.is_available():
            kwargs["gpu"] = -1
        kwargs = utils.load_yaml(kwargs)
        print(kwargs)
        utils.unflatten_dict(kwargs)
        self.flags = kwargs
        self.config = configs.GlobalConfig(**self.flags)

    def exp(self, name, datasets):
        exp_dict = utils.get_classes(EXPPATH, Experiment, pkg="mllib.exp")
        if "base_logdir" not in self.flags:
            baselogdir = self.config.base_logdir
        else:
            baselogdir = self.flags.pop("base_logdir")
        if "rel_logdir" not in self.flags:
            rellogdir = utils.gen_relative_logdir(f'{name}_{datasets}')
        else:
            rellogdir = self.flags.pop("rel_logdir")
        self.config.update({
            "logdir": os.path.join(baselogdir, rellogdir),
            "rel_logdir": rellogdir,
            "base_logdir": baselogdir
        })
        if self.config.print_config:
            print(self.config)
        global config
        config = self.config
        priv = self.config.private_file
        if priv is not None and priv:
            extras = utils.get_classes(priv, Experiment, pkg=None)
            for k in extras:
                if k in exp_dict:
                    print(f"WARNING: {k} is already a predefined experimment")
                else:
                    exp_dict[k] = extras[k]

        global experiment
        experiment = exp_dict[name](config=self.config, datasets=datasets)
        experiment()
        atexit.unregister(crash_save)

    def download(self, name, **kwargs):
        raise NotImplementedError()

    def extract(self, input_dir, out_file, preproc=None, name=None):
        raise NotImplementedError()

    def data(self, datasets, method=""):
        exp_dict = utils.get_classes(EXPPATH, Experiment)
        expr = exp_dict['data'](config=self.config, datasets=datasets)
        if method == "":
            call = expr
        else:
            call = getattr(expr, method)
        call()

    def available(self):
        exp_dict = utils.get_classes(EXPPATH, Experiment)
        print("available experiments are:")
        for k in exp_dict.keys():
            print(f'-- {k}')


def main():
    Fire(Main)
