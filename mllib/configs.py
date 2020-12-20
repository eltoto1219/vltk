import json
import os
from typing import List, Union

import yaml

from mllib.utils import (base_expirement_filename, get_most_free_gpu,
                         get_nvidia_gpu_memory)

DELIM = ","


class BaseConfig:
    _identify = None
    _overwritten = {}

    def __init__(self, **kwargs):
        for f, v in self:
            if f in kwargs:
                kv = kwargs.get(f)
                if v != kv:
                    setattr(self, f, kv)
                    self._overwritten[f] = v

    def __iter__(self):
        for k in self.__class__.__dict__:
            if k[0] != "_":
                yield k, getattr(self, k)

    def __str__(self):
        string = ""
        for k, v in self:
            if hasattr(v, "_identify"):
                string += f"{k}:\n"
                string += "".join([f"--{vsub}\n" for vsub in str(v).split("\n")])
            else:
                string += f"{k}:{v}\n"
        return string[:-1]

    @staticmethod
    def parse(arg):
        if isinstance(arg, str) and DELIM in arg:
            arg = arg.split(DELIM)
            if len(arg) == 0:
                arg = ""
            else:
                arg = tuple(arg)
        elif isinstance(arg, str) and arg.isdigit():
            return int(arg)
        elif isinstance(arg, str) and arg.lower() == "true":
            arg = True
        elif isinstance(arg, str) and arg.lower() == "false":
            arg = False
        return arg

    def to_dict(self):
        data = {}
        for k, v in self:
            if hasattr(v, "_identify"):
                data[k] = v.to_dict()
            else:
                data[k] = v
        return data

    def dump_json(self, file):
        json.dump(self.to_dict(), open(file, "w"))

    def dump_yaml(self, file):
        yaml.dump(self.to_dict(), open(file, "w"), default_flow_style=False)

    @classmethod
    def load(cls, fp_name_dict: Union[str, dict]):
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, config_dict):
        config = cls()
        config.update(config_dict)
        return config

    def update(self, updates: Union[dict, object]):
        if not isinstance(updates, dict):
            updates = updates.to_dict()
        for k, orig_v in self:
            if k in updates:
                v = updates.pop(k)
                if isinstance(v, dict) and hasattr(orig_v, "_identify"):
                    orig_v.update(v)
                else:
                    setattr(self, k, v)


class ExtractConfig(BaseConfig):
    outfile: str = ""
    inputdir: str = ""
    log_name: str = "extract_logs.txt"
    config_path: str = ""

    def __init__(self, outfile, inputdir, **kwargs):
        super().__init__(**kwargs)
        self.outfile = outfile
        self.inputdir = inputdir


class ModelConfig(BaseConfig):
    from_transformers: bool = True
    ckp_name_or_path: str = ""
    ckp_transformers: str = "unc-nlp/lxmert-base-uncased"
    load_epoch: int = 0
    known_labels: int = 1842
    r_layers: int = 5
    l_layers: int = 9
    x_layers: int = 5
    vit_variant: Union[str, None] = "ViT-B_16"


class PretrainConfig(BaseConfig):
    epochs: int = 4
    task_matched: bool = False
    task_mask_lm: bool = False
    task_obj_predict: bool = False
    visual_attr_loss: bool = False
    visual_obj_loss: bool = False
    visual_feat_loss: bool = False
    batch_size: int = 32


class EvalConfig(BaseConfig):
    half_precision: bool = True
    task_matched: bool = False
    task_mask_lm: bool = False
    task_obj_predict: bool = False
    visual_attr_loss: bool = False
    visual_obj_loss: bool = False
    visual_feat_loss: bool = False
    batch_size: int = 32


class TrainConfig(BaseConfig):
    learning_rate: float = 1e-5
    half_precision: bool = True
    epochs: int = 4
    gamma: float = 0.01
    max_norm: float = 5.0
    warmup: float = 0.10
    weight_decay: float = 0.01
    task_matched: bool = False
    task_mask_lm: bool = False
    task_obj_predict: bool = False
    visual_attr_loss: bool = False
    visual_obj_loss: bool = False
    visual_feat_loss: bool = False
    batch_size: int = 32


class DataConfig(BaseConfig):
    img_first: bool = False
    pad_collate: bool = True
    shuffle: bool = True
    num_workers: int = 8
    drop_last: bool = True
    pin_memory: bool = True
    sent_length: int = 20
    max_objects: int = 36
    img_max_size: int = 388
    patch_size: int = 16
    attribute_file: str = ""
    object_file: str = ""
    img_format: str = "jpg"
    percent_data: int = 1.0
    skip_eval: bool = False
    split: bool = "train"
    eval_split: bool = "eval"
    valid_split: bool = "valid"
    num_attrs: int = 400
    num_objects: int = 1600
    ignore_id: int = -100
    word_mask_rate: float = 0.15
    feature_mask_rate: float = 0.15
    random_feature_rate: float = 0.10
    random_word_rate: float = 0.10
    sentence_match_rate: float = 0.50
    truncate_sentence: bool = True
    return_token_type_ids: bool = True
    add_special_tokens: bool = True
    return_tensors: str = "pt"
    return_attention_mask: bool = True
    arrow_fields: Union[None, tuple, str] = None
    use_raw_imgs: bool = False
    pos_dim: int = 4


class PathesConfig(BaseConfig):
    datadirs: Union[List[str], str] = "/playpen1/home/avmendoz/data"
    vg_train: str = "vg/train"
    vg_test: str = "vg/test"
    vit_pretrained_dir = "vit/"
    vqa: str = "vqa/"
    gqa_val: str = "temp_gqa/train/valid.json"
    gqa_train: str = "temp_gqa/train/"
    gqa_test: str = ""
    gqa_testdev: str = "temp_gqa/testdev.json"
    gqa_labels: str = "temp_gqa/gqa_labels.json"
    vq_qa: str = "vg_qa/"
    vg_captions: str = "vg_captions/"
    coco_captions: str = "coco_captions/"
    coco_train_arrow: str = "arrow/coco_train2017.arrow"
    coco_valid_arrow: str = "arrow/coco_val2017.arrow"
    coco_test_arrow: str = "arrow/coco_test2017.arrow"
    vg_train_arrow: str = "arrow/vg_train.arrow"
    vg_test_arrow: str = "arrow/vg_test.arrow"
    temp_lxmert_answers: str = "labels/lxmert_answers.json"
    temp_lxmert_train: tuple = "lxmert_data/train/"
    temp_lxmert_eval: str = "lxmert_data/mscoco_minival.json"
    temp_lxmert_test: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for f, v in self:
            if f != "datadirs":
                if isinstance(self.datadirs, str):
                    v = os.path.join(self.datadirs, v)
                else:
                    for dd in self.datadirs:
                        temp_v = os.path.join(dd, v)
                        if ("." in v and os.path.isfile(temp_v)) or os.path.isdir(
                            temp_v
                        ):
                            v = temp_v
                if not (os.path.isfile(v) or os.path.isdir(v)):
                    v = None
                setattr(self, f, v)


class PrivateConfig(BaseConfig):
    pass


class ExperimentConfig(BaseConfig):
    pass


class GlobalConfig(BaseConfig):

    data: DataConfig = None
    model: ModelConfig = None
    pathes: PathesConfig = None
    evaluate: Union[None, EvalConfig] = None
    run: Union[None, PretrainConfig, TrainConfig, ExtractConfig, EvalConfig] = None
    private: PrivateConfig = None
    experiment: ExperimentConfig = None
    test_save: bool = False

    logging: bool = True
    logdir: str = os.path.join(os.environ.get("HOME", os.getcwd()), "logs")
    logfile: str = "logs.txt"
    output_dir: str = os.path.join(os.environ.get("HOME", ""), "outputs")
    command: str = None
    ckp_name: str = None
    gpu: int = None
    aux_gpu: int = None
    dryrun: bool = False
    seed: int = 1
    percent_min_gpu_free_mem: float = 0.75
    print_config: bool = True
    model_name: Union[None, str] = None
    datasets: Union[None, str] = None
    known_commands: tuple = ("train", "pretrain", "data", "eval", None)
    eval_aliases: tuple = ("testdev", "eval", "dev", "evaluation", "inference")
    train_aliases: tuple = ("train", "finetune", "pretrain")
    valid_aliases: tuple = ("val", "valid", "validation")
    test_aliases: tuple = ("test",)

    def __init__(self, command, **kwargs):
        super().__init__(**kwargs)
        assert command in self.known_commands
        self.command = command
        self._set_gpus()
        self._set_run(kwargs)
        self.data = DataConfig(**kwargs)
        self.model = ModelConfig(**kwargs)
        self.pathes = PathesConfig(**kwargs)
        self.private = PrivateConfig(**kwargs)
        self.experiment = ExperimentConfig(**kwargs)
        self.logfile = os.path.join(self.logdir, self.logfile)
        self.ckp_name = base_experiment_filename(self.output_dir, kwargs, self)
        if self.print_config:
            print(self)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    def _set_gpus(self):
        if self.gpu is not None and self.aux_gpu is not None:
            assert self.aux_gpu != self.gpu
        if self.gpu is None:
            self.gpu = get_most_free_gpu()
        if self.aux_gpu is None:
            set_aux = False
            gpus = get_nvidia_gpu_memory()
            for k in gpus:
                p_use = gpus[k][0] / gpus[k][1]
                if p_use < 1.0 - self.percent_min_gpu_free_mem and int(k) != self.gpu:
                    self.aux_gpu = int(k)
                    set_aux = True
                    break
            if not set_aux:
                if self.gpu != -1:
                    print("WARNING: all models using aux gpu will be on cpu")
                    print()
                self.aux_gpu = -1

    def _set_run(self, kwargs):
        if self.command == "pretrain":
            self.run = PretrainConfig(**kwargs)
        elif self.command == "train":
            self.run = TrainConfig(**kwargs)
        elif self.command == "extract":
            self.run = ExtractConfig(**kwargs)
        if self.command in ("data", "eval"):
            self.run = EvalConfig(**kwargs)
        if not kwargs.get("skip_eval", False) and self.command not in ("data", "eval"):
            self.evaluate = EvalConfig(**kwargs)
