import datetime
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import chain
from typing import Dict, List, Union

import torch
from vltk import COMPLEXPATH
from vltk.abc.visndatasetadapter import VisnDatasetAdapters
from vltk.abc.loop import Loop
from vltk.abc.visnlangdatasetadapter import VisnLangDatasetAdapters
from vltk.inspect import collect_args_to_func, get_classes
from vltk.modeling import Get as Mget
from vltk.modeling.configs import Get

__all__ = ["ComplexExperiment", "ComplexExpIdentifier", "ComplexExperiments"]

_visnlangdatasetadapters = VisnLangDatasetAdapters()
_visndatasetadapters = VisnDatasetAdapters()


class ComplexExperiments:
    def __init__(self):
        if "EXPDICT" not in globals():
            global COMPLEXDICT
            COMPLEXDICT = get_classes(
                COMPLEXPATH, ComplexExpIdentifier, pkg="vltk.complex"
            )

    def avail(self):
        return list(COMPLEXDICT.keys())

    def get(self, name):
        return COMPLEXDICT[name]

    def add(self, name, dset):
        COMPLEXDICT[name] = dset


class ComplexExpIdentifier:
    pass


class ComplexExperiment(ComplexExpIdentifier, ABC):
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
        self.logdir = getattr(self.config, "logdir", None)
        if self.logdir is not None and self.config.logging:
            os.makedirs(self.logdir, exist_ok=True)

        self._init_datasets()
        self._init_models()
        self._init_loops()
        self._init_gradient_tracking()
        self._experiment_outputs = {}

    @property
    def experiment_outputs(self):
        return self._experiment_outputs

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
                if name not in self.config.models.main_model:
                    model = model.to(torch.device(self.config.aux_gpu))
                else:
                    model = model.to(torch.device(self.config.gpu))
        # set all extra torch.nn.Modules to main gpu for now
        if self.extra_modules is not None:
            for name, nn in self.extra_modules.items():
                nn = nn.to(torch.device(self.config.gpu))

    def _init_gradient_tracking(self):
        # hardcoded now for time
        if self.model_dict:
            for name, model in self.model_dict.items():
                name = name.split("_")[0]
                conf = getattr(self.config.models, name)
                if hasattr(conf, "freeze_layers"):
                    layers_to_freeze = conf.freeze_layers
                    for n, p in model.named_parameters():
                        if "layer" in n.lower():
                            if layers_to_freeze.pop(0):
                                p.requires_grad = False

    def _init_datasets(self):
        # first check to see if any train or any val
        any_train = False
        any_val = False
        # check for train and eval loops
        for loop_key in self.loops_to_models:
            loop_cls = loop_key
            if loop_cls.is_train:
                any_train = True
            else:
                any_val = True

        self.label_to_id = {}
        self.dataset2splits = defaultdict(set)
        self.train_visnlangdatasetadapterdict = defaultdict(dict)
        self.eval_visnlangdatasetadapterdict = defaultdict(dict)
        self.train_visndatasetadapterdict = defaultdict(dict)
        self.eval_visndatasetadapterdict = defaultdict(dict)
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
                    and any_val
                ) or (
                    split in train_splits
                    and not self.config.data.skip_train
                    and any_train
                ):
                    visnlangdatasetadapter = _visnlangdatasetadapters.get(name).from_config(
                        self.config.data, splits=split
                    )[split]
                else:
                    continue
                for l in sorted(visnlangdatasetadapter.labels):
                    if l not in self.label_to_id:
                        self.label_to_id[l] = label_id
                        label_id += 1
                print(f"Added VisnLangDatasetAdapter {name}: {split}")
                if (
                    name in eval_datasets
                    and split in eval_splits
                    and not self.config.data.skip_eval
                    and any_val
                ):
                    self.eval_visnlangdatasetadapterdict[name][split] = visnlangdatasetadapter
                if (
                    split in train_splits
                    and not self.config.data.skip_train
                    and any_train
                ):
                    self.train_visnlangdatasetadapterdict[name][split] = visnlangdatasetadapter
                is_name, is_split = zip(*visnlangdatasetadapter.data_info[split].items())
                is_name = is_name[0]
                is_split = is_split[0][0]
                if self.config.data.extractor is not None:
                    is_path = visnlangdatasetadapter.get_arrow_split(
                        self.config.data.datadirs, is_split, self.config.data.extractor
                    )
                    visndatasetadapter = _visndatasetadapters.get(self.config.data.extractor).from_file(
                        is_path
                    )
                else:
                    visndatasetadapter = visnlangdatasetadapter.get_imgid_to_raw_path(
                        self.config.data.datadirs, is_split
                    )

                print(f"Added VisnDatasetAdapter {is_name}: {is_split}")

                if (
                    name in eval_datasets
                    and split in eval_splits
                    and not self.config.data.skip_eval
                    and any_val
                ):
                    self.eval_visndatasetadapterdict[is_name][is_split] = visndatasetadapter
                if (
                    split in train_splits
                    and not self.config.data.skip_train
                    and any_train
                ):
                    self.train_visndatasetadapterdict[is_name][is_split] = visndatasetadapter

    def _init_models(self):
        model_dict = {}
        models = set(chain(*list(self.loops_to_models.values())))
        for model in models:
            if not isinstance(model, str):
                raise Exception
            # get model class
            mclass = Mget[model]
            # get modl config
            mconfig = Get[model.split("_")[0]]

            # add model config to the vltk config object
            self.config.models.add(model.split("_")[0], mconfig)
            # get pointer to model config in vltk config
            config = getattr(self.config.models, model.split("_")[0])
            # instantiate model
            if getattr(config, "checkpoint", None) is not None and hasattr(
                mclass, "from_pretrained"
            ):
                model_instance = mclass.from_pretrained(
                    config.checkpoint, config=config
                )
            elif not hasattr(mclass, "from_pretrained"):
                # mandatory = false: ignore all required args. however we need to instead
                # get something as to where we get as many required args as are in the dict as possible
                arg_dict = collect_args_to_func(
                    mclass.__init__, config.to_dict(), mandatory=False
                )
                arg_dict["config"] = config
                model_instance = mclass(**arg_dict)
                if getattr(config, "checkpoint", None) is not None:
                    model_instance.load_state_dict(
                        torch.load(config.checkpoint), strict=False
                    )

            model_dict[model] = model_instance
        self._model_dict = model_dict

    def _init_loops(self):

        loop_dict = {}
        loop_info = {}
        for loop_key in self.loops_to_models:
            assert isinstance(loop_key, Loop)
            loop_cls = loop_key
            loop_name = loop_cls.name
            is_train = loop_cls.is_train
            if is_train:
                if self.config.data.skip_train:
                    continue
                visnlangdatasetadapterdict = self.train_visnlangdatasetadapterdict
                visndatasetadapterdict = self.train_visndatasetadapterdict
            else:
                visnlangdatasetadapterdict = self.eval_visnlangdatasetadapterdict
                visndatasetadapterdict = self.eval_visndatasetadapterdict
                if self.config.data.skip_eval or len(visnlangdatasetadapterdict) == 0:
                    continue

            loop = loop_cls(
                config=self.config,
                model_dict={
                    k: v
                    for k, v in self.model_dict.items()
                    if k in self.loops_to_models.get(loop_key)
                },
                extra_modules=self.extra_modules,
                datasets=self.datasets,
                visndatasetadapterdict=visndatasetadapterdict,
                visnlangdatasetadapterdict=visnlangdatasetadapterdict,
                label_dict=self.label_to_id,
            )

            if (loop.is_train and not self.config.data.skip_train) or (
                not loop.is_train and not self.config.data.skip_eval
            ):
                loop_dict[loop_name] = loop
                loop_info[loop_name] = loop.is_train

        print(f"Loaded Loops: {list(loop_dict.keys())}")

        self._loop_dict = loop_dict
        self._loop_info = loop_info
        order = sorted(
            list(loop_info.keys()), key=lambda x: int(loop_info[x]), reverse=True
        )
        self._order = order

    def write(self, epoch_output: dict = None):
        info = self.loginfo(epoch_output)
        if self.config.logging and info is not None and info:
            logfile = os.path.join(self.config.logdir, "log.txt")
            assert logfile is not None
            with open(logfile, "a") as f:
                date = datetime.datetime.now()
                f.write(f"Time: {date} \n {info} \n")
                f.flush()
            return True
        return False

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
        """
        TODO: FIX
        self.experiment_outputs.dump(save_name)
        """
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

            self.experiment_outputs[f"epoch_{epoch}"] = epoch_output
            self.write(loop_output)
            if self.config.test_save or (self.config.save_after_epoch and any_train):
                self.save()
        if (
            not self.config.save_after_epoch
            and not self.config.test_run
            and ((any_train and self.config.save_after_exp) or self.config.test_save)
        ):
            self.save()

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
