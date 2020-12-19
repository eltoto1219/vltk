import os
from dataclasses import dataclass
from typing import List, Union

from mllib.utils import (base_expirement_filename, get_most_free_gpu,
                         get_nvidia_gpu_memory)

DELIM = ","


class BaseConfig:
    def __init__(self, kwargs):
        for f in self.__dict__:
            str_field = f
            if str_field in kwargs:
                setattr(
                    self,
                    str_field,
                    BaseConfig.parse(kwargs.get(str_field)),
                )

    @staticmethod
    def parse(arg):
        if DELIM in arg:
            arg = arg.split(DELIM)
            if len(arg) == 0:
                arg = ""
            else:
                arg = tuple(arg)
        elif isinstance(arg, str) and arg.isdigit():
            return int(arg)
        return arg

    def json(self):
        raise NotImplementedError()

    def yaml(self):
        raise NotImplementedError()

    def dict(self):
        raise NotImplementedError()

    def dump(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()


class ExtractConfig:
    outfile: str
    inputdir: str
    log_name: str = "extract_logs.txt"
    config_path: str = ""

    def __init__(self, outfile, inputdir, **kwargs):
        self.outfile = outfile
        self.inputdir = inputdir
        for f in dir(self):
            str_field = f
            if str_field in kwargs:
                setattr(
                    self,
                    str_field,
                    BaseConfig.parse(kwargs.get(str_field)),
                )


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


@dataclass
class PathesConfig:
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

    def __init__(self, kwargs):
        for f in self.__dict__:
            str_field = f
            if str_field in kwargs:
                setattr(
                    self,
                    str_field,
                    BaseConfig.parse(kwargs.get(str_field)),
                )

        for f, v in self.__dict__.items():
            if f.name == "datadirs":
                continue
            f = f.name
            if isinstance(self.datadirs, str):
                v = os.path.join(self.datadirs, v)
            else:
                for dd in self.datadirs:
                    temp_v = os.path.join(dd, v)
                    if ("." in v and os.path.isfile(temp_v)) or os.path.isdir(temp_v):
                        v = temp_v
            if not (os.path.isfile(v) or os.path.isdir(v)):
                v = None
            setattr(self, f, v)


class PrivateConfig(BaseConfig):
    pass


@dataclass
class GlobalConfig:

    logpath: str
    command: str
    ckp_name: str
    gpu: int
    aux_gpu: int

    data: DataConfig
    model: ModelConfig
    pathes: PathesConfig
    evaluate: Union[None, EvalConfig]
    run: Union[None, PretrainConfig, TrainConfig, ExtractConfig, EvalConfig]
    private: PrivateConfig

    dryrun: bool = False
    logging: bool = True
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

    # log stuff
    log_dir: str = (
        os.path.join(os.environ.get("HOME"), "logs")
        if os.environ.get("HOME", False)
        else os.path.join(os.getcwd(), "logs")
    )
    log_file: str = "logs.txt"
    output_dir: str = "/playpen1/home/avmendoz/outputs"

    def print(self):
        print(" === Config === ")
        for k in self.__dict__:
            v = getattr(self, k, None)
            if "_" not in k[:2] and (
                type(v) == tuple
                or type(v) == int
                or type(v) == bool
                or type(v) == dict
                or type(v) == float
                or type(v) == str
            ):
                print(k, "=", v)
            elif "_" not in k[:2] and v is not None:
                print(k, ":")
                for j in v.__dict__:
                    f = getattr(v, j, None)
                    if "_" not in j[:2] and (
                        type(f) == tuple
                        or type(f) == dict
                        or type(f) == bool
                        or type(f) == int
                        or type(f) == float
                        or type(f) == str
                    ):
                        print("--", j, "=", f)
        print(" === ====== === ")
        print()

    def __init__(self, command, **kwargs):
        kwargs.pop("command", None)
        assert command in self.known_commands
        self.command = command
        for f in self.__dict__:
            str_field = f
            if str_field in kwargs:
                setattr(
                    self,
                    str_field,
                    BaseConfig.parse(kwargs.get(str_field)),
                )

        self.data = DataConfig(kwargs)
        self.model = ModelConfig(kwargs)
        self.pathes = PathesConfig(kwargs)
        self.private = PrivateConfig(kwargs)

        gpu = getattr(self, "gpu", None)
        aux_gpu = getattr(self, "aux_gpu", None)
        if gpu is not None and aux_gpu is not None:
            assert aux_gpu != gpu
        if gpu is None:
            self.gpu = get_most_free_gpu()
        if aux_gpu is None:
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

        evaluate = None
        run = None

        if self.command == "pretrain":
            run = PretrainConfig(kwargs)
        elif self.command == "train":
            run = TrainConfig(kwargs)
        elif self.command == "extract":
            run = ExtractConfig(kwargs)
        if self.command == "data" or self.command == "eval":
            run = EvalConfig(kwargs)
        if not self.data.skip_eval and self.command not in ("data", "eval"):
            evaluate = EvalConfig(kwargs)

        self.run = run
        self.evaluate = evaluate

        # other stuff
        self.logpath = os.path.join(self.log_dir, self.log_file)
        self.ckp_name = base_expirement_filename(self.output_dir, kwargs, self)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        self.dryrun

        self.print()
