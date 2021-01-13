import atexit
import sys
from io import StringIO
import random

import torch
from fire import Fire

from vltk import configs, main_functions, utils
from vltk.abc.experiment import Experiment
from vltk.maps import dirs

_experiments = dirs.Exps()
STDERR = sys.stderr = StringIO()


def crash_save():
    errorlog = STDERR.getvalue()
    if "experiment" in globals():
        experiment = globals()["experiment"]
        if experiment is not None and config is not None:
            save_on_crash = getattr(config, "save_on_crash", False)
            if config.email is not None:
                utils.send_email(config.email, errorlog)
            if save_on_crash:
                try:
                    experiment.save()
                    print("\nCRASH SAVE SUCCESS: vltk command crashed and was saved")
                except Exception:
                    print("\nFAILURE: vltk command crashed and was not saved")
            else:
                print("\nWARNING: vltk command crashed and was not saved")


@atexit.register
def restore_stdout():
    sys.stderr = sys.__stderr__
    sys.stderr.write(STDERR.getvalue())
    print()


class Main(object):
    """ class to handle cli arguments"""

    def __init__(self, **kwargs):
        if not torch.cuda.is_available():
            kwargs["gpu"] = -1
        kwargs = utils.unflatten_dict(kwargs)
        kwargs = utils.load_yaml(kwargs)
        self.flags = kwargs
        self.config = configs.Config(**self.flags)
        random.seed(self.config.seed)

    def exp(self, name, datasets):
        @atexit.register
        def inner_crash_save():
            return crash_save()

        global config
        config = self.config
        priv = self.config.private_file
        cls_dict = utils.get_classes(priv, Experiment, pkg=None)
        if priv is not None and priv:
            for name, clss in cls_dict.items():
                if name in _experiments.avail():
                    print(f"WARNING: {name} is already a predefined experiment")
                    _experiments.add(name, clss)
        main_functions.run_experiment(
            config, flags=self.flags, name=name, datasets=datasets
        )
        atexit.unregister(inner_crash_save)

    def download(self, name, **kwargs):
        raise NotImplementedError()

    def extract(
        self,
        extractor,
        dataset,
    ):
        extracted_data = main_functions.extract_data(
            extractor=extractor,
            dataset=dataset,
            config=self.config,
            features=features,
            image_preprocessor=image_preprocessor,
            img_format=img_format,
            flags=self.flags,
        )
        print(extracted_data)

    def data(self, datasets, method=""):
        expr = _experiments.get("data")(config=self.config, datasets=datasets)
        if method == "":
            call = expr
        else:
            call = getattr(expr, method)
        call()


def main():
    Fire(Main)
