import atexit
import os
import sys
from io import StringIO

import torch
from fire import Fire

from mllib import configs, experiments, utils

EXPRDICT = {'evallxmert': experiments.EvalLxmert, 'data': experiments.Data}
STDERR = sys.stderr = StringIO()


@atexit.register
def crash_save():
    errorlog = STDERR.getvalue()
    if experiment is not None and config is not None:
        save_on_crash = getattr(config, "save_on_crash", False)
        if config.email is not None and config.email_on_failure:
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
        utils.unflatten_dict(kwargs)
        self.flags = kwargs
        self.config = configs.GlobalConfig(**self.flags)

    def exp(self, name, datasets):
        logname = utils.logfile_name(f'{name}_{datasets}')
        if "logdir" not in self.flags:
            self.config.update({"logdir": os.path.join(self.config.logdir, logname)})
        print(f"\nUPDATED LOGDIR: {self.config.logdir}\n")
        global config
        config = self.config
        global experiment
        experiment = EXPRDICT[name](config=self.config, datasets=datasets)
        experiment()
        atexit.unregister(crash_save)

    def download(self, name, **kwargs):
        raise NotImplementedError()

    def extract(self, input_dir, out_file, preproc=None, name=None):
        raise NotImplementedError()

    def data(self, datasets, method=""):
        expr = EXPRDICT['data'](config=self.config, datasets=datasets)
        if method == "":
            call = expr
        else:
            call = getattr(expr, method)
        call()


def main():
    Fire(Main)
