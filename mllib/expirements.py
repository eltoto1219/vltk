import datetime
import os
from abc import ABC, abstractmethod
from typing import Dict

import torch

from mllib import loops, outputs
from mllib.models import factories


def get_loop(loop_name=None):
    loop_dict = {"evallxmertloop": loops.EvalLxmert}
    loop = loop_dict.get(loop_name, None)
    assert loop is not None, (loop_name, loop_dict.keys())
    return loop


class BaseExperiment(ABC):
    # for now lets just define dataset in loop
    def __init__(self, config, datasets):

        # maybe we can assume experiments are homegeneous within each loop
        # defualt stuff
        torch.manual_seed(self.config.seed)
        self.datasets = datasets
        self.config = config
        assert self.datasets is not None, "must specify `datasets` when firing command"
        self.epochs = self.config.run.epochs

        # experiment setup now
        # find someway to specify in the future
        loops_to_models = self.make_loops_to_models_dict()
        self.set_loops_to_models(loops_to_models)
        self.set_loops()
        self.set_model_gradient_tracking()
        self.set_logs()

    @abstractmethod
    def make_log_info(self, **kwargs) -> str:
        return ''

    @abstractmethod
    def make_loops_to_models_dict(self) -> dict:
        return {}

    @property
    def logdir(self):
        return getattr(self, "_logdir", None)

    @property
    def model_dict(self):
        return self._model_dict

    @property
    def loop_dict(self):
        return self._loop_dict

    @property
    def loop_info(self):
        return self._loop_dict

    @property
    def loops_to_models(self) -> dict:
        loops_to_models = getattr(self, "_loops_to_models", None)
        assert loops_to_models is not None, "must call `self.set_loops_to_models` before calling this function"
        return loops_to_models

    @property
    def extra_modules(self):
        assert hasattr(self, "_name_to_module")
        return self._name_to_module

    @property
    def loop_order(self):
        return self._order

    def set_extra_modules(self, name_to_module: dict):
        self._name_to_module = name_to_module

    def set_model_devices(self):
        for name, model in self.model_dict.items():
            if name == "vit":
                model = model.to(self.config.aux_gpu)
            else:
                model = model.to(self.config.gpu)

    def set_model_gradient_tracking(self):
        # hardcoded now for time
        for name, model in self.model_dict.items():
            if name == "vit":
                for n, p in model.named_parameters():
                    if "embed" not in n:
                        p.requires_grad = False
            else:
                pass

    def set_loops_to_models(self, loops_to_models: Dict[str, str]):
        self._loops_to_models = loops_to_models

    def set_loops(self):
        # limitation: cannot have same loop with different models
        model_dict = {
            model: factories.model_name_to_instance(
                model_name=model, config=self.config)
            for model in set(self.loops_to_models.items())
        }
        loop_dict = {
            loop: get_loop(loop)(
                config=self.config,
                models={
                    k: v for k, v in model_dict.items() if k in
                    self.loops_to_models(loop)
                },
                datasets=self.datasets
            ) for loop, model in self.loops_to_models.items()
        }
        self._model_dict = model_dict
        self._loop_dict = loop_dict
        self._loop_info = {k: {"is_train": v.is_train} for k, v in loop_dict.items()}
        order = []
        for name, info in self._loop_info.items():
            if info["is_train"]:
                order.insert(name, 0)
            else:
                order.insert(name, 1)
        self._order = order

    def set_log(self):
        logdir = getattr(self.config, "logdir", None) if self.config.log and not self.config.dryrun else None
        os.makedirs(logdir, exist_ok=True)

    def append_log(self, info: str = None):
        if self.log_file is not None and info is not None:
            with open(self.log_file, "a") as f:
                date = datetime.datetime.now()
                f.write(f"Time: {date} \n {info} \n")
                f.flush()
            return True
        return False

    def save(self, loop_name):
        if (self.config.dryrun and not self.config.test_save) or not self.loop_info[loop_name]["is_train"]:
            return False
        for name, model in self.model_dict.items():
            save_name = model + f"_{self.cur_epoch}.pt"
            save_name = os.path.join(self.logdir, save_name)
            torch.save(model, save_name)
        for extra, torch_module in self.extra_modules.items():
            save_name = extra + f"_{self.cur_epoch}.pt"
            save_name = os.path.join(self.logdir, save_name)
            torch.save(torch_module, save_name)
        save_name = os.path.join(self.logdir, "exp_outputs.pkl")
        self.experiment_outputs.dump()
        save_name = os.path.join(self.logdir, "config.yaml")
        self.config.to_yaml(save_name)

    def __call__(self):
        self.set_model_devices()
        self.experiment_outputs = outputs.ExperimentOutputs()
        for epoch in range(self.epochs):
            epoch_output = {}
            self.cur_epoch = epoch
            for loop_name in self.loop_order:
                loop = self.loop_dict.get(loop_name)
                loop_output = loop()
                epoch_output[loop_name] = loop_output
                self.append_log(self.make_log_info(**epoch_output))
                self.save()
                if len(self.loop_order) == 1 and not self.loop_info[loop_name]["is_train"]:
                    break
            self.experiment_outputs.add(epoch, epoch_output)


class EvalLxmert(BaseExperiment):
    def __init__(self, config, datasets):
        super().__init__(config, datasets)
        self.command = "eval"
        self.load()

    def make_log_info(**kwargs):
        '''
        kwargs will be  loop output from every run, with the run name being the key
        '''
        logstr = ''
        for k, v in kwargs.items():
            logstr += f'{k}: accuracy={v.accuracy} '
        return logstr

    def make_loops_to_models_dict(self):
        return {"evallxmertloop": ["lxmert"]}

    def load(self):
        # will need dir for multi-model experiments
        lxmert_ckp_path = self.config.model.ckp_name_or_path
        lxmert = self.model_dict["lxmert"]
        assert os.path.isfile(lxmert_ckp_path), lxmert_ckp_path
        lxmert.load_state_dict(torch.load(lxmert_ckp_path, map_location="cpu"))
