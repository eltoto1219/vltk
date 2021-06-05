import os
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Union

import torch
from vltk import LOOPPATH, 
# from vltk.dataset import UniversalLoader
from vltk.inspect import get_classes


class LoopIdentifier:
    pass


class Loop(LoopIdentifier, ABC):
    def __init__(
        self,
        config: object,
        datasets: str,
        label_dict: dict,
        visndatasetadapterdict: dict,
        visnlangdatasetadapterdict: dict,
        model_dict: Union[dict, None],
        extra_modules: Union[dict, None] = None,
    ):
        self.label_dict = label_dict
        self.model_dict = model_dict
        self.extra_modules = extra_modules
        self.config = config
        self.datasets = datasets
        self.cur_step = 0
        self.scheduler = None
        self.half_precision = getattr(config.train, "half_precision", False)
        self.main_device = (
            f"cuda:{config.gpu}" if getattr(config, "gpu", -1) != -1 else "cpu"
        )
        self.aux_model_device = (
            f"cuda:{config.aux_gpu}" if getattr(config, "aux_gpu", -1) != -1 else "cpu"
        )
        self.scaler = None if not self.half_precision else torch.cuda.amp.GradScaler()

        self._init_loader(visnlangdatasetadapterdict, visndatasetadapterdict, label_dict)
        self._init_models_and_extras(model_dict, extra_modules)
        self._init_optim()

        self._dataset = self.loader.dataset

        self.logdir = getattr(self.config, "logdir", None)
        if self.logdir is not None and self.config.logging:
            os.makedirs(self.logdir, exist_ok=True)

    def __key(self):
        name = self.is_name + "_"
        train = str(id(self.is_train)) + "_"
        if self.model_dict is None:
            models = ""
        else:
            models = "_".join(list(self.model_dict.keys())) + "_"
        if self.extra_modules is None:
            extras = ""
        else:
            extras = "_".join(list(self.extra_modules.keys())) + "_"
        return name + train + models + extras

    def __str__(self):
        return self.key()

    def __hash__(self):
        return hash(self.__key())

    # make MUCH better in the future
    def __eq__(self, other):
        if isinstance(other, Loop):
            if self.name == other.name and (
                (self.is_train and other.is_train)
                or (not self.is_train and not other.is_train)
            ):
                if self.extra_modules is None and other.extra_modules is None:
                    pass
                elif self.extra_modules is not None and other.extra_modules is None:
                    return False
                elif self.extra_modules is None and other.extra_modules is not None:
                    return False
                elif self.extra_modules.keys() != other.extra_modules.keys():
                    return False

                if self.model_dict is None:
                    if other.model_dict is None:
                        return True
                    else:
                        return False
                else:
                    if tuple(self.model_dict.keys()) == tuple(other.model_dict.keys()):
                        return True
                    else:
                        return False
                return True
            else:
                return False
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def _init_models_and_extras(self, model_dict, extra_modules=None):
        if model_dict is not None:
            for k, v in model_dict.items():
                v_old = getattr(self, k, None)
                assert v_old is None, (type(v_old), k)
                setattr(self, k, v)
        if extra_modules is not None:
            for k, v in extra_modules.items():
                v_old = getattr(self, k, None)
                assert v_old is None, type(v_old)
                v = v.to(self.main_device)
                setattr(self, k, v)

    @property
    def total_steps(self):
        if self.is_train:
            return self.config.train.epochs * len(self.loader)
        else:
            return len(self.loader)

    @property
    def warmup(self):
        return getattr(self, "_warmup", None)

    @property
    def optim(self):
        return getattr(self, "_optim", None)

    @property
    def dataset(self):
        return self._dataset

    def set_batch_size(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                self._bz = v.size(0)
                break

    def get_grad_params(self):
        parameters = []
        for k, v in self.model_dict.items():
            parameters.extend([p for p in v.parameters() if p.requires_grad])
        if self.extra_modules is not None:
            for k, v in self.extra_modules.items():
                parameters.extend([p for p in v.parameters() if p.requires_grad])
        return parameters

    def tqdm_update(self, info: dict = None):
        if info is not None:
            clean_info = {}
            for k, v in info.items():
                if v is None or (not isinstance(v, bool) and not v):
                    continue
                else:
                    clean_info[k] = v

            self._tqdm.set_postfix(**clean_info)

    def write_iter(self, info: dict = None):
        if info is not None:
            clean_info = {}
            for k, v in info.items():
                if v is None or (not isinstance(v, bool) and not v):
                    continue
                else:
                    clean_info[k] = v
        logstr = ""
        for k, v in info.items():
            logstr += f"{k}={v}; "

        if self.config.logging and info is not None and info:
            logfile = os.path.join(self.config.logdir, "cur_epoch.txt")
            assert logfile is not None
            with open(logfile, "a") as f:
                date = datetime.datetime.now()
                f.write(f"Time: {date} | {logstr} \n")
                f.flush()
            return True
        return False

    def toCuda(self, batch, device=None):
        self.loader.toCuda(batch, device)

    def toTrain(self):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.nn.Module):
                v.train()

    def toEval(self):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.nn.Module):
                v.eval()

    # move to support multiple optimizers in the future for even more loop generality
    def step(self, loss=None):
        if self.is_train and loss is not None:
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn..clip_grad_norm_(
                    self.get_grad_params(), self.config.train.max_norm
                )
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn..clip_grad_norm_(
                    self.get_grad_params(), self.config.train.max_norm
                )
                self.optim.step()

            self.warmup.step()

    def get_lr(self):
        if not self.is_train:
            return []
        elif self.warmup is not None:
            return self.warmup.get_lr()
        else:
            lrs = []
            for param_group in self.optim.param_groups:
                lrs.append(param_group["lr"])
            return lrs

    def __iter__(self):
        for x in self.loader:
            yield x

    def __call__(self) -> dict:
        loop_outputs = defaultdict(list)
        if self.is_train:
            self.toTrain()
        else:
            self.toEval()
        with torch.no_grad() if not self.is_train else .dummy_context():
            for batch in self.tqdm:
                if self.is_train:
                    self.optim.zero_grad()
                self.set_batch_size(batch)
                with self.forward_context():
                    model_outputs = self.forward(batch)
                outputs = self.loop(batch, model_outputs)
                self.write_iter(outputs)

                if self.is_train:
                    losses = outputs.get("losses", None)
                    assert losses is not None
                    self.step(losses)
                self.cur_step += 1
                if outputs is not None and loop_outputs is not None:
                    for k, v in outputs.items():
                        loop_outputs[k].append(v)
                else:
                    loop_outputs = None
            return loop_outputs

    @property
    def split(self):
        return self._split

    def get_visnlangdatasetadapter(self):
        return self.loader.dataset.visnlangdatasetadapterdict

    def get_visndatasetadapter(self):
        return self.loader.dataset.visndatasetadapterdict

    def _init_loader(self, visnlangdatasetadapterdict, visndatasetadapterdict, label_dict):
        datasets = self.datasets if self.is_train else self.config.data.eval_datasets
        self.loader = UniversalLoader(
            config=self.config.data,
            names=datasets,
            label_dict=label_dict,
            visndatasetadapterdict=visndatasetadapterdict,
            visnlangdatasetadapterdict=visnlangdatasetadapterdict,
        )

    @classmethod
    def eval_instance(cls, eval_name):
        eval_cls = type(eval_name, (cls,), {"is_train": False, "name": eval_name})
        assert not eval_cls.is_train, "loop is already in eval mode"
        return eval_cls

    @abstractmethod
    def loop(self, batch, model_outputs) -> dict:
        return None

    @abstractmethod
    def forward(self, batch) -> object:
        return None

    @property
    @abstractmethod
    def name(self) -> str:
        return ""

    @property
    @abstractmethod
    def is_train(self):
        pass
