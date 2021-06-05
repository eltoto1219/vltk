import atexit
import os
import random
import sys
from io import StringIO

import torch
from fire import Fire

from vltk import commands, configs
from vltk.utils import base
from vltk.abc.complex import ComplexExperiment, ComplexExperiments
from vltk.abc.simple import SimpleExperiment, SimpleExperiments
from vltk.inspect import get_classes

_simple_experiments = SimpleExperiments()
_complex_experiments = ComplexExperiments()
STDERR = sys.stderr = StringIO()


def crash_save():
    errorlog = STDERR.getvalue()
    config = globals()["config"]
    if config.email is not None:
        base.send_email(config.email, errorlog)

    # if "experiment" in globals():
    #     experiment = globals()["experiment"]
    #     if experiment is not None and config is not None:
    #         raise Exception("woot")
    #         save_on_crash = getattr(config, "save_on_crash", False)
    #         if config.email is not None:
    #             .send_email(config.email, errorlog)
    #         if save_on_crash:
    #             try:
    #                 experiment.save()
    #                 print("\nCRASH SAVE SUCCESS: vltk command crashed and was saved")
    #             except Exception:
    #                 print("\nFAILURE: vltk command crashed and was not saved")
    #         else:
    #             print("\nWARNING: vltk command crashed and was not saved")


def add_exps_from_dir(config):
    exp_dir = config.experimentdir
    sys.path.append(exp_dir)
    # now we need to add exp_dir to the path
    simple_experiments = get_classes(exp_dir, SimpleExperiment, pkg=None)
    for k, v in simple_experiments.items():
        if k in _simple_experiments.avail():
            print(f"WARNING: {k} is already a predefined simple experiment")
        _simple_experiments.add(v)
    experiments = get_classes(exp_dir, ComplexExperiment)
    for k, v in experiments.items():
        if k in _simple_experiments.avial():
            print(f"WARNING: {k} is already a predefined complex experiment")
        _complex_experiments.add(v)


@atexit.register
def restore_stdout():
    sys.stderr = sys.__stderr__
    sys.stderr.write(STDERR.getvalue())
    print()


class Main(object):
    """ class to handle cli arguments"""

    def __init__(self, **kwargs):
        if not torch.cuda.is_available():
            kwargs["gpu"] = "cpu"

        kwargs = .unflatten_dict(kwargs)
        kwargs = .load_yaml(kwargs)
        self.flags = kwargs
        if self.flags is None:
            self.flags = {}
        self.config = configs.Config(**self.flags)
        random.seed(self.config.seed)
        os.chdir(os.getcwd())

    def simple(self, name):
        @atexit.register
        def inner_crash_save():
            return crash_save()

        global config
        config = self.config

        priv = self.config.experimentdir
        if priv is not None:
            add_exps_from_dir(config)

        commands.run_simple_experiment(
            config,
            flags=self.flags,
            name_or_exp=name,
            datasets=self.config.data.train_datasets,
        )
        atexit.unregister(inner_crash_save)

    def exp(self, name):
        @atexit.register
        def inner_crash_save():
            return crash_save()

        global config
        config = self.config

        priv = self.config.experimentdir
        if priv is not None:
            add_exps_from_dir(config)

        commands.run_experiment(
            config,
            flags=self.flags,
            name_or_exp=name,
            datasets=self.config.data.train_datasets,
        )
        atexit.unregister(inner_crash_save)

    def download(self, name, **kwargs):
        raise NotImplementedError()

    def extract(
        self,
        extractor,
        dataset,
    ):

        extracted_data = commands.extract_data(
            extractor=extractor,
            dataset=dataset,
            config=self.config,
            flags=self.flags,
        )
        print(extracted_data)

    def data(self, datasets, method=""):

        datasets = configs.Config.handle_iterables(datasets)
        expr = _complex_experiments.get("data")(config=self.config, datasets=datasets)
        if method == "":
            call = expr
        else:
            call = getattr(expr, method)
        call()


def main():
    Fire(Main)
