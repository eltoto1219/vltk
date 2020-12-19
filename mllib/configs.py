import os
from dataclasses import dataclass, fields
from typing import Union

from mllib.utils import (base_expirement_filename, get_most_free_gpu,
                         get_nvidia_gpu_memory)


@dataclass
class ExtractConfig:
    le: str
    input_dir: str
    batch_size: int = 4
    log_name: str = "extract_logs.txt"
    config_path: str = ""

    def __init__(self, out_file, input_dir, **kwargs):

        for f in fields(self):
            str_field = f.name
            if str_field in kwargs:
                setattr(self, str_field, kwargs.get(str_field))


@dataclass
class ModelConfig:
    from_transformers: bool = True
    ckp_name_or_path: str = ""
    ckp_transformers: str = "unc-nlp/lxmert-base-uncased"
    load_epoch: int = 0
    known_labels: int = 1842
    r_layers: int = 5
    l_layers: int = 9
    x_layers: int = 5
    vit_variant: Union[str, None] = "ViT-B_16"

    def __init__(self, **kwargs):
        for f in fields(self):
            str_field = f.name
            if str_field in kwargs:
                setattr(self, str_field, kwargs.get(str_field))


@dataclass
class PretrainConfig:
    epochs: int = 4
    task_matched: bool = False
    task_mask_lm: bool = False
    task_obj_predict: bool = False
    visual_attr_loss: bool = False
    visual_obj_loss: bool = False
    visual_feat_loss: bool = False
    batch_size: int = 32

    def __init__(self, **kwargs):
        for f in fields(self):
            str_field = f.name
            if str_field in kwargs:
                setattr(self, str_field, kwargs.get(str_field))


@dataclass
class EvalConfig:
    half_percision: bool = True
    task_matched: bool = False
    task_mask_lm: bool = False
    task_obj_predict: bool = False
    visual_attr_loss: bool = False
    visual_obj_loss: bool = False
    visual_feat_loss: bool = False
    batch_size: int = 32

    def __init__(self, **kwargs):
        for f in fields(self):
            str_field = f.name
            if str_field in kwargs:
                setattr(self, str_field, kwargs.get(str_field))


@dataclass
class TrainConfig:
    learning_rate: float = 1e-5
    half_percision: bool = True
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

    def __init__(self, **kwargs):
        for f in fields(self):
            str_field = f.name
            if str_field in kwargs:
                setattr(self, str_field, kwargs.get(str_field))


@dataclass
class DataConfig:
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

    def __init__(self, **kwargs):
        for f in fields(self):
            str_field = f.name
            if str_field in kwargs:
                value = kwargs.get(str_field)
                if str_field == "arrow_fields":
                    value = value.replace("'", "")
                    value = value.replace('"', "")
                    value = value.replace(" ", "")
                    value = value.split(",")
                    if len(value) == 1 and value[0] == "":
                        value = ""
                    else:
                        value = tuple(value)
                        if "img_id" not in value:
                            value = ("img_id",) + value
                setattr(self, str_field, value)


@dataclass
class PathesConfig:
    data_dir: str = "/playpen1/home/avmendoz/data"
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
        for f in fields(self):
            str_field = f.name
            if str_field in kwargs:
                setattr(self, str_field, kwargs.get(str_field))


@dataclass
class GlobalConfig:

    logpath: str
    command: str
    ckp_name: str

    data: DataConfig
    model: ModelConfig
    pathes: PathesConfig
    evaluate: Union[None, EvalConfig] = None
    run: Union[None, PretrainConfig, TrainConfig, ExtractConfig, EvalConfig] = None

    dryrun: bool = False
    logging: bool = True
    gpu: int = None
    aux_gpu: int = None
    seed: int = 1
    percent_min_gpu_free_mem: float = 0.75
    print_config: bool = True
    model_name: Union[None, str] = None
    datasets: Union[None, str] = None
    valid_commands: tuple = ("train", "pretrain", "data", "eval")
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

    def __init__(self, command=None, **kwargs):

        assert command in self.valid_commands
        if "gpu" not in kwargs:
            self.gpu = get_most_free_gpu()

        self.command = command
        if command == "pretrain":
            self.run = PretrainConfig(**kwargs)
        elif command == "train":
            self.run = TrainConfig(**kwargs)
        elif command == "extract":
            self.run = ExtractConfig(**kwargs)
        elif command == "data" or command == "eval":
            kwargs["skip_eval"] = True
            self.run = EvalConfig(**kwargs)
        if self.command not in ("data", "eval") and not kwargs.get("skip_eval", False):
            self.evaluate = EvalConfig(**kwargs)

        self.data = DataConfig(**kwargs)
        self.model = ModelConfig(**kwargs)
        self.pathes = PathesConfig(**kwargs)

        # set other sub attrs
        for f in fields(self):
            str_field = f.name
            if str_field in kwargs:
                setattr(self, str_field, kwargs.get(str_field))

        # auto detect aux_gpu
        if "aux_gpu" not in kwargs:
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
        else:
            assert self.gpu != kwargs["aux_gpu"]

        # other stuff
        self.logpath = os.path.join(self.log_dir, self.log_file)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        self.ckp_name = base_expirement_filename(self.output_dir, kwargs, self)
        if self.print_config:

            print(" === Config === ")
            for k, v in self.__dict__.items():
                if "Config" in type(v).__name__:
                    print(k, ":")
                    for f in fields(v):
                        str_field = f.name
                        val = getattr(v, str_field)
                        print("--", str_field, "=", val)
                else:
                    print(k, "=", v)
            print(" === ====== === ")
            print()
