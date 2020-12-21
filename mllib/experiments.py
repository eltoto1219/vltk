import datetime
import os
import random
from abc import ABC, abstractmethod
from itertools import chain
from typing import Dict, List

import torch

from mllib import data, loops, outputs
from mllib.models import factories


def get_loop(loop_name=None):
    loop_dict = {
        "evallxmert": loops.EvalLxmert,
        "data": loops.Data,
        "evalvitlxmert": loops.EvalViTLxmert,
        "trainvitlxmert": loops.TrainViTLxmert
    }
    loop = loop_dict.get(loop_name, None)
    assert loop is not None, (loop_name, loop_dict.keys())
    return loop


class BaseExperiment(ABC):
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
        self.epochs = self.config.run.epochs
        self.main_device = f"cuda:{config.gpu}" if config.gpu != -1 else "cpu"
        self.aux_model_device = f"cuda:{config.aux_gpu}" if config.gpu != -1 else "cpu"
        self.logdir = getattr(self.config, "logdir", None)
        if self.logdir is not None and self.config.logging:
            os.makedirs(self.logdir, exist_ok=True)

        self.set_loops()
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

    def set_loops(self):
        model_dict = {
            model: factories.model_name_to_instance(
                model_name=model, config=self.config)
            for model in set(chain(*list(self.loops_to_models.values()))) if model is not None
        }
        loop_dict = {
            loop: get_loop(loop)(
                config=self.config,
                model_dict={
                    k: v for k, v in model_dict.items() if k in
                    self.loops_to_models.get(loop)
                },
                extra_modules=self.extra_modules,
                datasets=self.datasets
            ) for loop, model in self.loops_to_models.items()
        }
        self._model_dict = model_dict
        self._loop_dict = loop_dict
        self._loop_info = {k: {"is_train": v.is_train} for k, v in loop_dict.items()}
        order = []
        for name, info in self._loop_info.items():
            if info["is_train"]:
                order.insert(0, name)
            else:
                order.insert(-1, name)
        self._order = order

    def write(self, info: str = None):
        if self.config.logging and info is not None and info:
            logfile = getattr(self.config, "logfile", None)
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
            "warmups": {}
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
            if (any_train or self.config.test_save) and self.config.save_after_epoch:
                self.save()
        if (
            (any_train or self.config.test_save)
            and (self.config.save_after_exp and not self.config.save_after_epoch)
        ):
            self.save()

    @property
    @abstractmethod
    def name(self) -> str:
        return ''

    @property
    @abstractmethod
    def loops_to_models(self) -> Dict[str, List[str]]:
        return {}

    @abstractmethod
    def loginfo(self, **kwargs) -> str:
        return ''


class Data(BaseExperiment):
    name: str = "data"
    loops_to_models: dict = {"data": [None]}

    def loginfo(self):
        return super().loginfo()

    def keys(self):
        entry = None
        for loop_name, loop in self:
            for x in loop:
                entry = x
                for k, v in entry.items():
                    shape = None
                    if isinstance(v, torch.Tensor):
                        shape = v.shape
                    print(k, type(v), shape)
                break
            break

    def transpose(self):
        assert self.config.data.img_first
        assert not self.config.data.arrow_fields
        for loop_name, loop in self:
            for x in loop:
                entry = x
                for k, v in entry.items():
                    shape = None
                    if isinstance(v, torch.Tensor):
                        shape = v.shape
                    print(k, type(v), shape)
                data.UniversalDataset.transpose_img2txt(entry, img_keys=["raw_imgs", "raw_sizes"], device="cpu")
                for k, v in entry.items():
                    shape = None
                    if isinstance(v, torch.Tensor):
                        shape = v.shape
                    print(k, type(v), shape)
                break


class EvalLxmert(BaseExperiment):

    name: str = "evallxmert"
    loops_to_models: dict = {"evallxmert": ["lxmert"]}
    # extra_modules = {"connector": torch.nn.Linear(18464, 2048)}

    def __init__(self, config, datasets):
        super().__init__(config, datasets)
        self.load()

    def loginfo(self, **kwargs):
        '''
        kwargs will be  loop output from every run, with the run name being the key
        '''
        logstr = ''
        for k, v in kwargs.items():
            logstr += f'{k}: accuracy={v.accuracy} '
        return logstr

    # move loading to abstract class when more time
    def load(self):
        # will need dir for multi-model experiments
        lxmert_ckp_path = self.config.models.lxmert.ckp_name_or_path
        if lxmert_ckp_path is not None:
            lxmert = self.model_dict["lxmert"]
            assert os.path.isfile(lxmert_ckp_path), "must specify valid checkpoint"
            lxmert.load_state_dict(torch.load(lxmert_ckp_path, map_location="cpu"))


class TrainViTLxmert(BaseExperiment):

    name: str = "trainvitlxmert"
    loops_to_models: dict = {"evalvitlxmert": ["lxmert", "vit"], "trainvitlxmert": ["lxmert", "vit"]}
    extra_modules = {"connector": torch.nn.Linear(18464, 2048)}

    def __init__(self, config, datasets):
        super().__init__(config, datasets)

    def loginfo(self, **kwargs):
        '''
        kwargs will be  loop output from every run, with the run name being the key
        '''
        logstr = ''
        for k, v in kwargs.items():
            logstr += f'{k}: accuracy={v.accuracy} '
        return logstr
