import datetime
import os
import random
from abc import ABC, abstractmethod
from itertools import chain
from typing import Dict, List, Union

import torch
from vltk import factory, outputs
from vltk.maps import dirs
from vltk.utils import IdentifierClass

__all__ = ["Experiment"]

_loop = dirs.Loops()



class Experiment(IdentifierClass, ABC):
    # for now lets just define dataset in loop
    def __init__(self, config, datasets):

        # maybe we can assume experiments are homegeneous within each loop
        # defualt stuff
        self.cur_epoch = 0
        self.datasets = datasets
        self.config = config
        self.seed = self.config.seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        assert self.datasets is not None, "must specify 'datasets' when firing command"
        self.epochs = self.config.train.epochs
        self.main_device = f"cuda:{config.gpu}" if config.gpu != -1 else "cpu"
        self.aux_model_device = f"cuda:{config.aux_gpu}" if config.gpu != -1 else "cpu"
        self.logdir = getattr(self.config, "logdir", None)
        if self.logdir is not None and self.config.logging:
            os.makedirs(self.logdir, exist_ok=True)

        self._init_loops()
        self.set_model_gradient_tracking()
        exp_info = self.get_exp_info()
        self.experiment_outputs = outputs.ExperimentOutputs(**exp_info)

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
    def loop_order(self):
        return self._order

    @property
    def extra_modules(self):
        return None

    def get_model(self):
        pass

    def set_model_devices(self):
        if self.model_dict:
            for name, model in self.model_dict.items():
                if name in self.config.models.aux_models:
                    model = model.to(self.aux_model_device)
                else:
                    model = model.to(self.main_device)
        # set all extra torch.nn.Modules to main gpu for now
        if self.extra_modules is not None:
            for name, nn in self.extra_modules.items():
                nn = nn.to(self.main_device)

    def set_model_gradient_tracking(self):
        # hardcoded now for time
        if self.model_dict:
            for name, model in self.model_dict.items():
                if name == "vit":
                    for n, p in model.named_parameters():
                        if "embed" not in n:
                            p.requires_grad = False
                else:
                    pass

    def _init_loops(self):
        # move to auto detect models in the future
        model_dict = {
            model: factory.model_name_to_instance(model_name=model, config=self.config)
            for model in set(chain(*list(self.loops_to_models.values())))
            if model is not None
        }
        loop_dict = {}
        loop_info = {}
        first_loop_name = None
        for loop_key, model_list in self.loops_to_models.items():
            if isinstance(loop_key, str):
                loop_cls = _loop.get(loop_key)
            else:
                loop_cls = loop_key

            loop_name = loop_cls.name

            if first_loop_name is not None:
                first_loop = loop_dict[first_loop_name]
                imagesetdict = first_loop.get_imageset()
                textsetdict = first_loop.get_textsetdict()
            else:
                first_loop_name = loop_name
                imagesetdict = None
                textsetdict = None
            loop = loop_cls(
                config=self.config,
                model_dict={
                    k: v
                    for k, v in model_dict.items()
                    if k in self.loops_to_models.get(loop_key)
                },
                extra_modules=self.extra_modules,
                datasets=self.datasets,
                imagesetdict=imagesetdict,
                textsetdict=textsetdict,
            )

            if (loop.is_train and not self.config.data.skip_train) or (
                not loop.is_train and not self.config.data.skip_eval
            ):
                loop_dict[loop_name] = loop
                loop_info[loop_name] = loop.is_train
        print(f"Loaded Loops: {list(loop_dict.keys())}")

        self._model_dict = model_dict
        self._loop_dict = loop_dict
        self._loop_info = loop_info
        order = sorted(
            list(loop_info.keys()), key=lambda x: int(loop_info[x]), reverse=True
        )
        self._order = order

    def write(self, info: str = None):
        if self.config.logging and info is not None and info:
            logfile = os.path.join(self.config.logdir, "log.txt")
            assert logfile is not None
            with open(logfile, "a") as f:
                date = datetime.datetime.now()
                f.write(f"Time: {date} \n {info} \n")
                f.flush()
            return True
        return False

    def save(self):
        print("\nsaving...\n")
        if self.model_dict:
            for name, model in self.model_dict.items():
                save_name = name + f"_{self.cur_epoch}.pt"
                save_name = os.path.join(self.config.logdir, save_name)
                torch.save(model.state_dict(), save_name)
        if self.extra_modules is not None:
            for extra, torch_module in self.extra_modules.items():
                save_name = extra + f"_{self.cur_epoch}.pt"
                save_name = os.path.join(self.config.logdir, save_name)
                torch.save(torch_module.state_dict(), save_name)
        optim_dict = {}
        save_name = f"optims_{self.cur_epoch}.pt"
        save_name = os.path.join(self.config.logdir, save_name)
        for loop_name, loop in self:
            if loop.is_train:
                optim_dict[loop_name] = loop.optim.state_dict()
        torch.save(optim_dict, save_name)
        save_name = os.path.join(self.config.logdir, "exp_outputs.pkl")
        self.experiment_outputs.dump(save_name)
        save_name = os.path.join(self.config.logdir, "config.yaml")
        self.config.dump_yaml(save_name)

    def get_exp_info(self):
        exp_info = {
            "name": self.name,
            "datasets": self.datasets,
            "loops_to_models": self.loops_to_models,
            "cur_steps": {},
            "cur_epoch": self.cur_epoch,
            "epochs": self.epochs,
            "total_steps": {},
            "schedulers": {},
            "warmups": {},
        }
        for loop_name, loop in self:
            exp_info["cur_steps"][loop_name] = loop.cur_step
            exp_info["total_steps"][loop_name] = loop.total_steps
            if loop.is_train and loop.warmup is not None:
                exp_info["warmups"][loop_name] = loop.warmup.state_dict()

        return exp_info

    def __iter__(self):
        for loop_name in self.loop_order:
            yield loop_name, self.loop_dict.get(loop_name)

    def __call__(self):
        print()
        self.set_model_devices()
        for epoch in range(self.epochs):
            epoch_output = {}
            self.cur_epoch = epoch
            any_train = False
            for i, (loop_name, loop) in enumerate(self):
                loop_output = loop()
                epoch_output[loop_name] = loop_output
                if loop.is_train:
                    any_train = True

                if len(self.loop_order) - 1 == i:
                    break

            exp_info = self.get_exp_info()
            self.experiment_outputs.add(epoch, epoch_output, **exp_info)
            self.write(self.loginfo(**epoch_output))
            if self.config.test_save or (self.config.save_after_epoch and any_train):
                self.save()
        if not self.config.save_after_epoch and (
            (any_train and self.config.save_after_exp) or self.config.test_save
        ):
            self.save()
            print("or here")

    @property
    @abstractmethod
    def name(self) -> str:
        return ""

    @property
    @abstractmethod
    def loops_to_models(self) -> Dict[Union[str, object], List[str]]:
        return {}

    @abstractmethod
    def loginfo(self, **kwargs) -> str:
        return ""
