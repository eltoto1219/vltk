# we can excpet list of forward fucntions at some other point
# for now, all we really need to care about is one forward method
# and how to convert that to eval if no class

import datetime
import gc
import json
import os
import random
import sys
from abc import ABC, abstractmethod
from collections import Iterable, OrderedDict, defaultdict
from copy import deepcopy
from statistics import mean
from typing import Dict, List, Union

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from vltk import SIMPLEPATH, 
from vltk.abc.visndatasetadapter import VisnDatasetAdapters
from vltk.abc.visnlangdatasetadapter import VisnLangDatasetAdapters
# from vltk.dataset import UniversalLoader
from vltk.inspect import get_classes
from vltk.loader.builder import init_datasets
from vltk.modeling import Get as Mget
from vltk.modeling.configs import Get

__all__ = ["SimpleExperiment", "SimpleIdentifier", "SimpleExperiments"]
_visnlangdatasetadapters = VisnLangDatasetAdapters()
_visndatasetadapters = VisnDatasetAdapters()

# find all packages and classes when pointed to a specific directory?


class SimpleExperiments:
    def __init__(self):
        if "SIMPLEDICT" not in globals():
            global SIMPLEDICT
            SIMPLEDICT = get_classes(SIMPLEPATH, SimpleIdentifier, pkg="vltk.simple")

    def avail(self):
        return list(SIMPLEDICT.keys())

    def get(self, name):
        return SIMPLEDICT[name]

    def add(self, experiment):
        name = experiment.__name__
        SIMPLEDICT[name] = experiment


class SimpleIdentifier:
    pass


class SimpleExperiment(SimpleIdentifier, ABC):
    cur_epoch: int = 1
    cur_step: int = 1

    def __init__(self, config, datasets):
        self.config = config
        self.datasets = datasets
        self.__currently_training = False
        self.epochs = self.config.train.epochs
        self.cur_epoch = 1
        self._init_dirs()
        self._init_seed()
        self._init_scaler()
        # self._init_datasets()
        self._init_loaders()
        self._init_models()
        self._init_optim()
        self._init_checkpoint()

        # do we benchmark?
        if not self.config.empty_cache:
            torch.backends.benchmark = True

    def garbage_collect(self):
        if self.config.empty_cache:
            gc.collect()
            # torch.cuda.empty_cache()

    def currently_training(self):
        return True if self.__currently_training == "train" else False

    def _init_checkpoint(self):
        checkpoint_dir = self.config.vltk_checkpoint_dir
        if checkpoint_dir is None or not checkpoint_dir:
            return
        highest_model_epoch = {}
        # we can think about making sure the label size is right later
        for f in os.listdir(checkpoint_dir):
            path = os.path.join(checkpoint_dir, f)
            if "info" in path:
                exp_outputs = json.load(open(path))
                if "scheduler" in exp_outputs:
                    self.scheduler.load_state_dict(exp_outputs["scheduler"])
                self.cur_epoch = exp_outputs["cur_epoch"]
                self.epochs = exp_outputs["epochs"]
                self.cur_step = exp_outputs["cur_steps"]
            if "_epoch_" in path:
                if len(tuple(f.split(".")[0].split("_"))) == 3:
                    model_n, _, epoch_n = tuple(f.split(".")[0].split("_"))

                    highest_model_epoch[model_n] = max(
                        highest_model_epoch.get(model_n, 0), int(epoch_n)
                    )

        # load the model weights
        for model_n, highest_epoch in highest_model_epoch.items():
            f = os.path.join(checkpoint_dir, f"{model_n}_epoch_{highest_epoch}.pt")
            model = getattr(self, model_n)
            print(f"reloading weights for {model_n} for epoch {highest_epoch}")

            checkpoint = torch.load(f)
            keys = deepcopy(list(checkpoint.keys()))
            for k in keys:
                k_stripped = k.replace("module.", "")
                checkpoint[k_stripped] = checkpoint.pop(k)
            model.load_state_dict(checkpoint)
            # setattr(self, model_n, model)

    def _init_scaler(self):
        # all devices must be on some gpu if we want to use scaler

        half_precision = getattr(self.config.train, "half_precision", False)
        self.half_precision = half_precision
        if self.config.gpu == "cpu":
            half_precision = False
        self.scaler = None if not half_precision else torch.cuda.amp.GradScaler()

    def _init_models(self):
        model_list = self.model_list
        model_dict = {}
        model_configs = {}
        state_dict = None
        default_device = self.config.gpu if torch.cuda.is_available() else -1

        for x in model_list:
            if isinstance(x, tuple):
                name = x[0]
                model_class = x[1]
            elif isinstance(x, str):
                name = x
                model_class = Mget[name]

            model_config = getattr(self.config.models, name, None)
            model_device = getattr(model_config, "device", default_device)
            setattr(self, f"name_{model_device}", model_device)
            # TODO:  add a preinitialized option

            checkpoint = getattr(model_config, "checkpoint", None)
            print(f"instantiating {name} from {checkpoint}")

            # CASE 1
            if (
                checkpoint is not None
                and hasattr(model_class, "from_pretrained")
                and not os.path.isfile(checkpoint)
            ):
                # this is a huggingface model, so the config must be added appropriately
                model_instance = model_class.from_pretrained(
                    checkpoint,
                )
                checkpoint = model_instance.state_dict()
                model_config = Get[name](**model_config.to_dict())
                model_instance = model_class(model_config)

            # CASE 2
            elif not hasattr(model_class, "from_pretrained"):
                model_instance = model_class(**model_config.to_dict())
                if checkpoint is not None:
                    state_dict = torch.load(checkpoint)
            # CASE 3
            else:
                try:
                    model_instance = model_class(model_config)
                except Exception:
                    model_instance = model_class(**model_config.to_dict())

            # CASE 4: try loading checkpoint one last time
            if checkpoint is not None and state_dict is not None:

                # 1. filter out unnecessary keys
                new_state_dict = OrderedDict(
                    {
                        k: v
                        for k, v in state_dict.items()
                        if k in model_instance.state_dict()
                        and v.shape
                        == model_instance.state_dict()
                        .get(k, torch.tensor([], requires_grad=False))
                        .shape
                    }
                )
                # 3. load the new state dict
                model_instance.load_state_dict(new_state_dict, strict=False)

            # for question answering models, we will need to resize the number of labels
            # accordingly
            if hasattr(model_instance, "resize_num_qa_labels"):
                assert (
                    getattr(self, "label_to_id", None) is not None
                ), "no label dict found"
                print(f"Number of Labels: {len(self.label_to_id)}")
                label_file = self.config.data.labels
                if label_file is not None or "":
                    model_instance.resize_num_qa_labels(
                        len(json.load(open(label_file)))
                    )
                else:
                    model_instance.resize_num_qa_labels(len(self.label_to_id))

            # SET DEVICE
            if isinstance(model_device, int):
                model_instance.to(torch.device(model_device))
            elif isinstance(model_device, list) and len(model_device) > 1:
                os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
                model_instance.to(torch.device(model_device[0]))
                model_instance = nn.DataParallel(
                    model_instance, device_ids=model_device
                ).to(torch.device(model_device[0]))
            elif isinstance(model_device, list) and len(model_device) == 1:
                model_instance.to(torch.device(model_device[0]))
            setattr(self, f"{name}_dev", model_device)

            model_dict[name] = model_instance
            model_configs[name] = model_config
            self.init_grad(name, model_instance)
            setattr(self, name, model_instance)

        self._model_dict = model_dict
        self._model_configs = model_configs

    # def _init_loader(self, visnlangdatasetadapterdict, visndatasetadapterdict, label_dict, train=True):
    #     datasets = self.datasets if train else self.config.data.eval_datasets
    #     loader = UniversalLoader(
    #         config=self.config.data,
    #         names=datasets,
    #         label_dict=label_dict,
    #         visndatasetadapterdict=visndatasetadapterdict,
    #         visnlangdatasetadapterdict=visnlangdatasetadapterdict,
    #     )
    #     # get necessary methods from the loader
    #     self.flatten_text = loader.dataset.flatten_text
    #     return loader

    @property
    def is_train(self):
        return getattr(self, "any_train", None)

    def _init_loaders(self):
        loaders, any_train, answer_to_id, object_to_id = init_datasets(self.config.data)
        self._loaders = loaders
        self.answer_to_id = answer_to_id
        self.object_to_id = object_to_id
        self.any_train = any_train
        # ttsd = self.train_visnlangdatasetadapterdict
        # tisd = self.train_visndatasetadapterdict
        # etsd = self.eval_visnlangdatasetadapterdict
        # eisd = self.eval_visndatasetadapterdict
        # l2id = self.label_to_id

        # loaders = {
        #     "train": self._init_loader(ttsd, tisd, l2id) if ttsd else None,
        #     "eval": self._init_loader(etsd, eisd, l2id) if etsd else None,
        # }
        # self._loaders = [(k, v) for k, v in loaders.items()]
        # self.is_train = any(map(lambda x: x == "train", [k for k in loaders]))
        # self._loaders = sorted(self._loaders, key=lambda x: x[0], reverse=True)
        # for k, v in loaders.items():
        #     if v is not None:
        #         self.transpose_img2txt = v.dataset.transpose_img2txt

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
        if self.is_train and self.model_dict:
            parameters = []
            for k, v in self.model_dict.items():
                parameters.extend(v.parameters())
            assert parameters, "no parameters added to optimizer"
            self._optim = AdamW(
                parameters,
                lr=self.config.train.learning_rate,
                weight_decay=self.config.train.weight_decay,
            )
            total = self.total_steps
            n_steps = int(total * self.config.train.warmup)
            self._scheduler = get_linear_schedule_with_warmup(
                self._optim, num_warmup_steps=n_steps, num_training_steps=total
            )

    @property
    def forward_context(self):
        if self.scaler is None:
            return .dummy_context
        else:
            return torch.cuda.amp.autocast

    @property
    def batch_size(self):
        if getattr(self, "_bz", None) is not None:
            return self._bz
        else:
            if self.currently_training():
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

    def _save_outputs(self, save_outputs):
        save_name = os.path.join(
            self.config.logdir, f"user_saved_epoch_{self.cur_epoch}.json"
        )
        json.dump(save_outputs, open(save_name, "w"))

    # def _init_datasets(self):
    #     # first check to see if any train or any val
    #     # check for train and eval loops
    #     self.label_to_id = {}
    #     self.dataset2splits = defaultdict(set)
    #     self.train_visnlangdatasetadapterdict = defaultdict(dict)
    #     self.eval_visnlangdatasetadapterdict = defaultdict(dict)
    #     self.train_visndatasetadapterdict = defaultdict(dict)
    #     self.eval_visndatasetadapterdict = defaultdict(dict)
    #     if not self.config.data.annotations:
    #         self.annotationdict = None
    #     else:
    #         self.annotationdict = {}

    #     train_datasets = set()
    #     eval_datasets = set()
    #     train_splits = set()
    #     eval_splits = set()

    #     # loop through train datasets
    #     for (dset, splits) in self.datasets:
    #         train_datasets.add(dset)
    #         train_splits = train_splits.union(splits)
    #         self.dataset2splits[dset] = self.dataset2splits[dset].union(splits)
    #     # then loop thorugh eval datasets
    #     for (dset, splits) in self.config.data.eval_datasets:
    #         assert (
    #             dset in train_datasets
    #         ), "eval datasets must also be present in train datasets"
    #         eval_datasets.add(dset)
    #         eval_splits = eval_splits.union(splits)
    #         if not splits.intersection(self.dataset2splits[dset]):
    #             self.dataset2splits[dset] = self.dataset2splits[dset].union(splits)

    #     label_id = 0
    #     for name in sorted(set(self.dataset2splits.keys())):
    #         for split in sorted(set(self.dataset2splits[name])):
    #             if (
    #                 name in eval_datasets
    #                 and split in eval_splits
    #                 and not self.config.data.skip_eval
    #             ) or (split in train_splits and not self.config.data.skip_train):
    #                 visnlangdatasetadapter = _visnlangdatasetadapters.get(name).from_config(
    #                     self.config.data, splits=split
    #                 )[split]
    #             else:
    #                 continue
    #             for l in sorted(visnlangdatasetadapter.labels):
    #                 if l not in self.label_to_id:
    #                     self.label_to_id[l] = label_id
    #                     label_id += 1
    #             print(f"Added VisnLangDatasetAdapter {name}: {split}")
    #             if (
    #                 name in eval_datasets
    #                 and split in eval_splits
    #                 and not self.config.data.skip_eval
    #             ):
    #                 self.eval_visnlangdatasetadapterdict[name][split] = visnlangdatasetadapter
    #             if split in train_splits and not self.config.data.skip_train:
    #                 self.train_visnlangdatasetadapterdict[name][split] = visnlangdatasetadapter
    #             is_name, is_split = zip(*visnlangdatasetadapter.data_info[split].items())
    #             is_name = is_name[0]
    #             is_split = is_split[0][0]
    #             if self.config.data.extractor is not None:
    #                 is_path = visnlangdatasetadapter.get_arrow_split(
    #                     self.config.data.datadirs, is_split, self.config.data.extractor
    #                 )
    #                 visndatasetadapter = _visndatasetadapters.get(self.config.data.extractor).from_file(
    #                     is_path
    #                 )
    #             else:
    #                 visndatasetadapter = visnlangdatasetadapter.get_imgid_to_raw_path(
    #                     self.config.data.datadirs, is_split
    #                 )
    #             if self.config.data.annotations and is_name not in self.annotationdict:
    #                 # TODO: must fix this later
    #                 path = os.path.join(
    #                     searchdirs[-1], is_name, "annotations/annotations.arrow"
    #                 )
    #                 ### THIS IS WEHRE I COME BACK TOO ###
    #                 self.eval_visndatasetadapterdict[is_name][is_split] = visndatasetadapter

    #             print(f"Added VisnDatasetAdapter {is_name}: {is_split}")

    #             if (
    #                 name in eval_datasets
    #                 and split in eval_splits
    #                 and not self.config.data.skip_eval
    #             ):
    #                 self.eval_visndatasetadapterdict[is_name][is_split] = visndatasetadapter
    #             if split in train_splits and not self.config.data.skip_train:
    #                 self.train_visndatasetadapterdict[is_name][is_split] = visndatasetadapter

    #     label_file = self.config.data.labels
    #     if label_file is not None or "":
    #         self.label_to_id = json.load(open(label_file))

    # vanilla mehtods
    def write_epoch(self, info: dict = None):
        logstr = ""
        for k, v in info.items():
            logstr += f"{k}={v}; "
        if self.config.logging and info is not None and info:
            logfile = os.path.join(self.config.logdir, "epoch_log.txt")
            if self.currently_training():
                desc = "train"
            else:
                desc = "eval"
            if os.path.isfile(logfile):
                open_type = "a"
            else:
                open_type = "w"
            with open(logfile, open_type) as f:
                date = datetime.datetime.now()
                out_str = f"{desc} | epoch: {self.cur_epoch} | date: {date} | {info} \n"
                f.write(out_str)
                f.flush()
            return True
        return False

    def write_iter(self, info: dict = None, inner_step=0):
        logstr = ""
        for k, v in info.items():
            logstr += f"{k}={v}; "
        if self.config.logging and info is not None and info:
            logfile = os.path.join(self.config.logdir, "steps_log.json")
            assert logfile is not None
            if self.currently_training():
                desc = "train"
            else:
                desc = "eval"
            open_type = "a" if self.cur_step > 1 else "w"
            with open(logfile, open_type) as f:

                date = datetime.datetime.now()
                json.dump(
                    {
                        "desc": desc,
                        "step": self.cur_step,
                        "data": info,
                        "time": str(date),
                    },
                    f,
                )
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
        save_name = os.path.join(self.config.logdir, "info.json")
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
            "name": type(self).__name__,
            "datasets": self.datasets,
            "cur_steps": self.cur_step,
            "cur_epoch": self.cur_epoch,
            "epochs": self.config.train.epochs,
        }

        if getattr(self, "scheduler", None) is not None:
            exp_info["scheduler"]: self.scheduler.state_dict()

        return exp_info

    # dunder methods

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
        for _ in range(self.epochs):
            if self.cur_epoch == self.epochs + 1:
                return
            # iterate through train and eval loops
            for (run, loader) in self.loaders:
                if loader is None:
                    continue
                else:
                    loop_output = self.inner_loop(
                        loader, train=run, epoch=self.cur_epoch
                    )
                    self.write_epoch(self._clean_dict(loop_output))
            # collect epoch output from each  loop
            if self.config.test_save or self.config.save_after_epoch:
                self.save()

            self.cur_epoch += 1

        if (
            not self.config.save_after_epoch
            and not self.config.test_run
            and self.config.save_after_exp
        ):
            self.save()

    def inner_loop(self, loader, train="train", epoch=None):
        self.__currently_training = train
        desc = "train" if self.currently_training() else "eval"
        # account for cache batch
        if (
            self.config.test_run
            and getattr(loader.dataset, "cache_batch_exists", False)
            and not self.config.data.overwrite_cache_batch
            and self.config.break_loop_on_test
        ):
            loader = [torch.load(loader.dataset.cache_batch_path)]

        _tqdm = tqdm(
            loader,
            desc=f"{desc}_{self.cur_epoch}/{self.config.train.epochs}",
            ncols=0,
            file=sys.stdout,
        )
        loop_outputs = defaultdict(list)
        save_outputs = defaultdict(list)
        if self.currently_training():
            self.toTrain()
        else:
            self.toEval()
        with torch.no_grad() if not self.currently_training() else .dummy_context():
            for inner_step, batch in enumerate(_tqdm):
                if self.currently_training() and self.model_dict:
                    self.optim.zero_grad()
                self.set_batch_size(batch)

                with self.forward_context():
                    forward_outputs = self.forward(batch)
                    assert isinstance(
                        forward_outputs, dict
                    ), "forward method must return dict"

                outputs = self._clean_dict(
                    self.iter_tqdm(forward_outputs, train=self.currently_training())
                )

                # something is wrong with this
                # save_outputs = self._clean_dict(
                #     self.iter_save(forward_outputs, train=train)
                # )
                temp_save_outputs = outputs
                _tqdm.set_postfix(epch=self.cur_epoch, **outputs)

                self.write_iter(outputs, inner_step)

                if self.currently_training():
                    losses = forward_outputs.get("losses", None)
                    if losses is None:
                        pass
                    else:
                        self.step(losses)
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
                # free empty space
                self.garbage_collect()
                if (
                    not self.config.empty_cache
                    and inner_step == 0
                    and self.cur_epoch == 0
                ):
                    pass
                    # torch.cuda.empty_cache()
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

    def step(self, loss=None):
        if loss is not None and self.model_dict:
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

            self.scheduler.step()

    def get_lr(self):
        if not self.is_train:
            return []
        elif self.scheduler is not None:
            return self.scheduler.get_lr()
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
    def scheduler(self):
        return getattr(self, "_scheduler", None)

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

    def toCuda(batch, device):
        # will retun batch, but is still in place)
        return .change_device(batch, device)

    def init_grad(name, model):
        pass

    # abstract methods

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
