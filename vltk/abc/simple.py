# experiments are composed of loops
# we can excpet list of forward fucntions at some other point
# for now, all we really need to care about is one forward method
# and how to convert that to eval if no class

import datetime
import json
import os
import random
import sys
from abc import ABC, abstractmethod
from collections import Iterable, defaultdict
from statistics import mean
from typing import Dict, List, Union

import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from vltk import SIMPLEPATH, utils
from vltk.abc.imageset import Imagesets
from vltk.abc.textset import Textsets
from vltk.dataset import UniversalLoader
from vltk.modeling import Get as Mget
from vltk.modeling.configs import Get
from vltk.utils import get_classes

__all__ = ["SimpleExperiment", "SimpleIdentifier", "SimpleExperiments"]
_textsets = Textsets()
_imagesets = Imagesets()


def collection_of_torch(x):
    pass


def collection_of_numpy(x):
    pass


class SimpleExperiments:
    def __init__(self):
        if "SIMPLEDICT" not in globals():
            global SIMPLEDICT
            SIMPLEDICT = get_classes(SIMPLEPATH, SimpleIdentifier, pkg="vltk.simple")

    def avail(self):
        return list(SIMPLEDICT.keys())

    def get(self, name):
        return SIMPLEDICT[name]

    def add(self, name, dset):
        SIMPLEDICT[name] = dset


class SimpleIdentifier:
    pass


class SimpleExperiment(SimpleIdentifier, ABC):
    cur_epoch: int = 0
    cur_step: int = 0

    def __init__(self, config, datasets):
        self.config = config
        self.datasets = datasets
        self._init_dirs()
        self._init_seed()
        self._init_scaler()
        self._init_datasets()
        self._init_loaders()
        self._init_models()
        self._init_optim()
        self._init_gradient_tracking()
        self.open_type = "w" if self.cur_epoch == 0 else "a"

    # hidden methods

    def _init_scaler(self):

        half_precision = getattr(self.config.train, "half_precision", False)
        if self.config.gpu == "cpu":
            half_precision = False
        self.scaler = None if not half_precision else torch.cuda.amp.GradScaler()

    def _init_models(self):
        model_list = self.model_list
        model_dict = {}
        model_configs = {}
        state_dict = None
        for x in model_list:
            if isinstance(x, tuple):
                name = x[0]
                model_class = x[1]
            elif isinstance(x, str):
                name = x
                model_class = Mget[name]

            model_config = getattr(self.config.models, name, None)
            if model_config is None:
                print(f"No Model Config For {name}", "returning class, not instance")
                model_dict[name] = model_class
                model_configs[name] = None
                setattr(self, name, model_class)
                continue
            checkpoint = getattr(model_config, "checkpoint", None)
            print(f"instantiating {name} from {checkpoint}")
            # load from checkpoint if specificed

            if checkpoint is not None and hasattr(model_class, "from_pretrained"):
                # this is a huggingface model, so the config must be added appropriately
                model_instance = model_class.from_pretrained(
                    checkpoint,
                )
                checkpoint = model_instance.state_dict()
                model_config = Get[name](**model_config.to_dict())
                model_instance = model_class(model_config)

            elif not hasattr(model_class, "from_pretrained"):
                model_instance = model_class(**model_config.to_dict())
                if checkpoint is not None:
                    state_dict = torch.load(checkpoint)
            else:
                # model does not have checkpoint
                try:
                    print("a")
                    model_instance = model_class(model_config)
                    print("b")
                except Exception:
                    print("c")
                    model_instance = model_class(**model_config.to_dict())
                    print("d")

            if checkpoint is not None and state_dict is not None:
                model_instance.load_state_dict(state_dict, strict=False)

            # for question answering models, we will need to resize the number of labels
            # accordingly
            if hasattr(model_instance, "resize_num_qa_labels"):
                assert (
                    getattr(self, "label_to_id", None) is not None
                ), "no label dict found"
                print(f"Number of Labels: {len(self.label_to_id)}")
                model_instance.resize_num_qa_labels(len(self.label_to_id))
            model_dict[name] = model_instance
            model_configs[name] = model_config
            setattr(self, name, model_instance)

        self._model_dict = model_dict
        self._model_configs = model_configs

    def _init_loader(self, textsetdict, imagesetdict, label_dict, train=True):
        datasets = self.datasets if train else self.config.data.eval_datasets
        loader = UniversalLoader(
            config=self.config.data,
            names=datasets,
            label_dict=label_dict,
            imagesetdict=imagesetdict,
            textsetdict=textsetdict,
        )
        return loader

    def _init_loaders(self):
        ttsd = self.train_textsetdict
        tisd = self.train_imagesetdict
        etsd = self.eval_textsetdict
        eisd = self.eval_imagesetdict
        l2id = self.label_to_id

        loaders = {
            "train": self._init_loader(ttsd, tisd, l2id) if ttsd else None,
            "eval": self._init_loader(etsd, eisd, l2id) if etsd else None,
        }
        self._loaders = [(k, v) for k, v in loaders.items()]
        self.is_train = any(map(lambda x: x == "train", [k for k in loaders]))
        self._loaders = sorted(self._loaders, key=lambda x: x[0], reverse=True)
        for k, v in loaders.items():
            if v is not None:
                self.transpose_img2txt = v.dataset.transpose_img2txt

    def _clean_dict(self, info):
        clean_info = {}
        for k, v in info.items():
            if (
                v is None
                or isinstance(v, torch.Tensor)
                or (
                    isinstance(v, Iterable)
                    and len(v) > 1
                    and isinstance(v[0], torch.Tensor)
                )
                or (not isinstance(v, bool) and not v)
            ):
                continue
            else:
                clean_info[k] = v

        return clean_info

    # other methods, too many methods
    def _init_optim(self):
        self.scheduler = None
        if self.is_train:
            parameters = []
            for k, v in self.model_dict.items():
                parameters.extend(v.parameters())
            assert parameters, "no parameters added to optimizer"
            self._optim = AdamW(
                parameters,
                lr=self.config.train.learning_rate,
                weight_decay=self.config.train.weight_decay,
            )
            if self.config.train.warmup == 0.0:
                self._warmup = None
            total = self.total_steps
            n_steps = int(total * self.config.train.warmup)
            self._warmup = get_linear_schedule_with_warmup(
                self._optim, num_warmup_steps=n_steps, num_training_steps=total
            )

    @property
    def forward_context(self):
        if self.scaler is None:
            return utils.dummy_context
        else:
            return torch.cuda.amp.autocast

    @property
    def batch_size(self):
        if getattr(self, "_bz", None) is not None:
            return self._bz
        else:
            if self.is_train:
                return self.config.train.batch_size
            else:
                return self.config.evaluate.batch_size

    # init methods

    def _init_dirs(self):
        if self.config.logdir is not None and self.config.logging:
            os.makedirs(self.config.logdir, exist_ok=True)

    def _init_seed(self):
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def _init_gradient_tracking(self):
        if self.model_dict:
            for name, model in self.model_dict.items():
                name = name.split("_")[0]
                conf = getattr(self.config.models, name, None)
                if conf is None:
                    continue
                prev_layer = None
                layer = 0
                if hasattr(conf, "freeze_layers"):
                    layers_to_freeze = conf.freeze_layers
                    for n, p in model.named_parameters():
                        if (
                            "blocks" in n.lower()
                            and prev_layer is not None
                            and str(prev_layer) in n
                        ):

                            print(f"{name}: freezing {n}")

                            p.requires_grad = False
                        elif "blocks" in n.lower() and str(layer) in n:
                            prev_layer = layer
                            if layers_to_freeze.pop(0):
                                print(f"{name}: freezing {n}")
                                p.requires_grad = False
                            layer += 1

    def _save_outputs(self, save_outputs):
        save_name = os.path.join(
            self.config.logdir, f"save_outputs_epoch_{self.cur_epoch}.pt"
        )
        json.dump(save_outputs, open(save_name, "w"))

    def _init_datasets(self):
        # first check to see if any train or any val
        # check for train and eval loops
        self.label_to_id = {}
        self.dataset2splits = defaultdict(set)
        self.train_textsetdict = defaultdict(dict)
        self.eval_textsetdict = defaultdict(dict)
        self.train_imagesetdict = defaultdict(dict)
        self.eval_imagesetdict = defaultdict(dict)
        train_datasets = set()
        eval_datasets = set()
        train_splits = set()
        eval_splits = set()

        # loop through train datasets
        for (dset, splits) in self.datasets:
            train_datasets.add(dset)
            train_splits = train_splits.union(splits)
            self.dataset2splits[dset] = self.dataset2splits[dset].union(splits)
        # then loop thorugh eval datasets
        for (dset, splits) in self.config.data.eval_datasets:
            assert (
                dset in train_datasets
            ), "eval datasets must also be present in train datasets"
            eval_datasets.add(dset)
            eval_splits = eval_splits.union(splits)
            if not splits.intersection(self.dataset2splits[dset]):
                self.dataset2splits[dset] = self.dataset2splits[dset].union(splits)

        label_id = 0
        for name in sorted(set(self.dataset2splits.keys())):
            for split in sorted(set(self.dataset2splits[name])):
                if (
                    name in eval_datasets
                    and split in eval_splits
                    and not self.config.data.skip_eval
                ) or (split in train_splits and not self.config.data.skip_train):
                    textset = _textsets.get(name).from_config(
                        self.config.data, splits=split
                    )[split]
                else:
                    continue
                for l in sorted(textset.labels):
                    if l not in self.label_to_id:
                        self.label_to_id[l] = label_id
                        label_id += 1
                print(f"Added Textset {name}: {split}")
                if (
                    name in eval_datasets
                    and split in eval_splits
                    and not self.config.data.skip_eval
                ):
                    self.eval_textsetdict[name][split] = textset
                if split in train_splits and not self.config.data.skip_train:
                    self.train_textsetdict[name][split] = textset
                is_name, is_split = zip(*textset.data_info[split].items())
                is_name = is_name[0]
                is_split = is_split[0][0]
                if self.config.data.extractor is not None:
                    is_path = textset.get_arrow_split(
                        self.config.data.datadirs, is_split, self.config.data.extractor
                    )
                    imageset = _imagesets.get(self.config.data.extractor).from_file(
                        is_path
                    )
                else:
                    imageset = textset.get_imgid_to_raw_path(
                        self.config.data.datadirs, is_split
                    )

                print(f"Added Imageset {is_name}: {is_split}")

                if (
                    name in eval_datasets
                    and split in eval_splits
                    and not self.config.data.skip_eval
                ):
                    self.eval_imagesetdict[is_name][is_split] = imageset
                if split in train_splits and not self.config.data.skip_train:
                    self.train_imagesetdict[is_name][is_split] = imageset

    # vanilla mehtods
    def write_epoch(self, info: dict = None):
        logstr = ""
        for k, v in info.items():
            logstr += f"{k}={v}; "
        if self.config.logging and info is not None and info:
            logfile = os.path.join(self.config.logdir, "log.txt")
            with open(logfile, self.open_type) as f:
                date = datetime.datetime.now()
                f.write(f"Time: {date} \n {info} \n")
                f.flush()
            return True
        return False

    def write_iter(self, info: dict = None):
        logstr = ""
        for k, v in info.items():
            logstr += f"{k}={v}; "
        if self.config.logging and info is not None and info:
            logfile = os.path.join(self.config.logdir, "cur_epoch.txt")
            assert logfile is not None
            with open(logfile, self.open_type) as f:
                date = datetime.datetime.now()
                f.write(f"Time: {date} \n {info} \n")
                f.flush()
            return True
        return False

    def save(self):
        print("\nsaving...\n")
        if self.model_dict:
            for name, model in self.model_dict.items():
                save_name = name + f"_epoch_{self.cur_epoch}.pt"
                save_name = os.path.join(self.config.logdir, save_name)
                torch.save(model.state_dict(), save_name)
        save_name = f"optim_epoch_{self.cur_epoch}.pt"
        if getattr(self, "optim", None) is not None:
            save_name = os.path.join(self.config.logdir, save_name)
            torch.save(self.optim.state_dict(), save_name)
        save_name = os.path.join(self.config.logdir, "exp_outputs.pkl")
        json.dump(self.get_exp_info(), open(save_name, "w"))
        save_name = os.path.join(self.config.logdir, "config.yaml")
        self.config.dump_yaml(save_name)
        if hasattr(self, "label_to_id"):
            json.dump(
                self.label_to_id,
                open(os.path.join(self.config.logdir, "labels.json"), "w"),
            )

    def get_exp_info(self):
        exp_info = {
            "name": self.name,
            "datasets": self.datasets,
            "cur_steps": self.cur_step,
            "cur_epoch": self.cur_epoch,
            "epochs": self.config.train.epochs,
        }

        if getattr(self, "warmup", None) is not None:
            exp_info["warmup"] = self.warmup.state_dict()
        if getattr(self, "scheduler", None) is not None:
            exp_info["scheduler"]: self.scheduler.state_dict()

        return exp_info

    # dunder methods

    def __iter__(self):
        for loop_name in self.loop_order:
            yield loop_name, self.loop_dict.get(loop_name)

    def __call__(self):
        self.outer_loop()

    # helper medhods

    def toTrain(self):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.nn.Module):
                v.train()

    def toEval(self):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.nn.Module):
                v.eval()

    # loop methods

    def outer_loop(self, epoch=None):
        for epoch in range(self.config.train.epochs):
            self.cur_epoch = epoch
            # iterate through train and eval loops
            for (run, loader) in self.loaders:
                if loader is None:
                    continue
                else:
                    loop_output = self.inner_loop(loader, train=run, epoch=epoch)
                    self.write_epoch(self._clean_dict(loop_output))
            # collect epoch output from each  loop
            if self.config.test_save or self.config.save_after_epoch:
                self.save()
        if (
            not self.config.save_after_epoch
            and not self.config.test_run
            and ((self.config.save_after_exp) or self.config.test_save)
        ):
            self.save()

        pass

    def inner_loop(self, loader, train=True, epoch=None):
        desc = "train" if train else "eval"
        # account for cache batch
        if (
            self.config.test_run
            and loader.dataset.cache_batch_exists
            and not self.config.data.overwrite_cache_batch
            and self.config.break_loop_on_test
        ):
            loader = [torch.load(loader.dataset.cache_batch_path)]

        _tqdm = tqdm(loader, desc=desc, ncols=0, file=sys.stdout)
        loop_outputs = defaultdict(list)
        save_outputs = defaultdict(list)
        if train:
            self.toTrain()
        else:
            self.toEval()
        with torch.no_grad() if not train else utils.dummy_context():
            for batch in _tqdm:

                if train and self.model_dict:
                    self.optim.zero_grad()
                self.set_batch_size(batch)
                with self.forward_context():
                    forward_outputs = self.forward(batch)
                    assert isinstance(
                        forward_outputs, dict
                    ), "forward method must return dict"

                outputs = self._clean_dict(self.iter_tqdm(forward_outputs, train=train))
                # something is wrong with this
                # save_outputs = self._clean_dict(
                #     self.iter_save(forward_outputs, train=train)
                # )
                temp_save_outputs = outputs
                _tqdm.set_postfix(epch=self.cur_epoch, **outputs)
                self.write_iter(outputs)

                if train:
                    losses = forward_outputs.get("losses", None)
                    if losses is None:
                        pass
                    else:
                        self.step(losses, train)
                self.cur_step += 1
                # handle loop outputs
                if outputs is not None and loop_outputs is not None:
                    for k, v in outputs.items():
                        loop_outputs[k].append(v)
                else:
                    loop_outputs = None
                # handle saveoutputs
                if temp_save_outputs is not None:
                    for k, v in temp_save_outputs.items():
                        save_outputs[k].append(v)
                # break, possibly if test
                if self.config.test_run and self.config.break_loop_on_test:
                    break

            if save_outputs:
                self._save_outputs(save_outputs)
            # try to take the mean of what we can from the loopoutputs
            for k, v in loop_outputs.items():
                try:
                    loop_outputs[k] = mean(map(lambda x: float(x), v))
                except Exception:
                    pass
            return loop_outputs

        pass

    def step(self, loss=None, train=True):
        if train and loss is not None and self.model_dict:
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(
                    self.get_grad_params(), self.config.train.max_norm
                )
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
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

    @property
    def model_configs(self):
        return self._model_dict

    @property
    def model_dict(self):
        return self._model_dict

    @property
    def loaders(self):
        return self._loaders

    @property
    def warmup(self):
        return getattr(self, "_warmup", None)

    @property
    def optim(self):
        return getattr(self, "_optim", None)

    @property
    def total_steps(self):
        if self.is_train:
            total_steps = 0
            for (run, l) in self.loaders:
                if run == "train" and l is not None:
                    total_steps += len(l)

            return self.config.train.epochs * total_steps
        else:
            return len(self.loader)

    def set_batch_size(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                self._bz = v.size(0)
                break

    def get_grad_params(self):
        parameters = []
        for k, v in self.model_dict.items():
            parameters.extend([p for p in v.parameters() if p.requires_grad])
        return parameters

    # abstract methods

    def toCuda(
        self,
        x: Union[
            Dict[str, Union[List[torch.Tensor], torch.Tensor]],
            torch.Tensor,
            List[torch.Tensor],
        ],
        device,
    ):

        # should do something recursive here, not sure what I am thinking
        if isinstance(x, torch.Tensor):
            return x.to(torch.device(device))
        elif isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(torch.device(device))
                else:
                    if isinstance(v[0], torch.Tensor):
                        x[k] = [
                            i.to(torch.device(device))
                            if isinstance(i, torch.Tensor)
                            else i
                            for i in v
                        ]
                    elif isinstance(v[0], str):
                        pass

            return x
        elif isinstance(x, list):
            return [i.to(torch.device(device)) for i in v]
        else:
            raise Exception("could not change the device")

    @property
    @abstractmethod
    def name(self):
        """
        define the name of the experiment
        """

    @abstractmethod
    def forward(self, batch) -> dict:
        """
        here the user defines the forward run of the model
        returns a dictionary of model outputs
        """

    @abstractmethod
    def iter_tqdm(self, iter_outputs, train=True):
        """
        here the user defines how they would like to configure a dictionary
        that will subesequently be processed and displayed on tqdm
        (this will also be saved to a temp log file)
        """

    @abstractmethod
    def iter_save(iter_outputs, train=True):
        """
        here the user defines what they would like to save from each iteraction
        they can conditionally save if train is true of false
        """

    @abstractmethod
    def epoch_logstr(self, loop_outputs, train=True):
        """
        here the user defines how they would like to configure a dictionary
        that will subesequently be processed and written to a log
        """

    @property
    @abstractmethod
    def model_list(self):
        """
        user defines a list of models. each list item is either a string that references
        a model available in the  library, or a tuple containing the model name and then the model class
        """
