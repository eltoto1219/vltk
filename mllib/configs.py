import os
from dataclasses import dataclass, fields
from typing import Union


@dataclass
class ROIFeaturesFRCNN:

    out_file: str
    input_dir: str
    batch_size: int = 4
    log_name: str = "extract_logs.txt"
    config_path: str = ""

    def __init__(self, out_file, input_dir, **kwargs):
        self.out_file = out_file
        self.input_dir = input_dir
        for f in fields(self):
            str_field = f.name
            if str_field in kwargs:
                setattr(self, str_field, kwargs.pop(str_field))


@dataclass
class GlobalConfig:

    log_dir: str = (
        os.path.join(os.environ.get("HOME"), "logs")
        if os.environ.get("HOME", False)
        else os.path.join(os.getcwd(), "logs")
    )
    log_file: str = "logs.txt"
    data_dir: str = "/playpen1/home/avmendoz/data"
    output_dir: str = "/playpen1/home/avmendoz/outputs"
    gpus: int = 1
    seed: int = 1

    def __init__(self, **kwargs):
        for f in fields(self):
            str_field = f.name
            if str_field in kwargs:
                setattr(self, str_field, kwargs.pop(str_field))
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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
    vit_variant: Union[str, None] = "ViT-B_32"

    def __init__(self, **kwargs):
        for f in fields(self):
            str_field = f.name
            if str_field in kwargs:
                setattr(self, str_field, kwargs.pop(str_field))


@dataclass
class PretrainConfig:
    train_batch_size: int = 32
    eval_batch_size: int = 64

    def __init__(self, **kwargs):
        for f in fields(self):
            str_field = f.name
            if str_field in kwargs:
                setattr(self, str_field, kwargs.pop(str_field))


@dataclass
class TrainConfig:
    epochs: int = 4
    task_matched: bool = False
    task_mask_lm: bool = False
    task_obj_predict: bool = False
    visual_attr_loss: bool = False
    visual_obj_loss: bool = False
    visual_feat_loss: bool = False
    train_batch_size: int = 32
    eval_batch_size: int = 64
    test_batch_size: int = 64
    dry_run: bool = True

    def __init__(self, **kwargs):
        for f in fields(self):
            str_field = f.name
            if str_field in kwargs:
                setattr(self, str_field, kwargs.pop(str_field))


@dataclass
class DataConfig:
    sent_length: int = 20
    max_objects: int = 36
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

    def __init__(self, **kwargs):
        for f in fields(self):
            str_field = f.name
            if str_field in kwargs:
                value = kwargs.pop(str_field)
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
class LoaderConfig:
    shuffle: bool = True
    num_workers: int = 8
    drop_last: bool = True
    pin_memory: bool = True

    def __init__(self, **kwargs):
        for f in fields(self):
            str_field = f.name
            if str_field in kwargs:
                setattr(self, str_field, kwargs.pop(str_field))


@dataclass
class PathesConfig:
    vg_tensor_train: str = "vg_raw/train"
    vg_tensor_test: str = "vg_raw/test"
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
    vg_train_arrow: str = "arrow/vg.arrow"
    vg_valid_arrow: str = "arrow/vg.arrow"
    vg_test_arrow: str = "arrow/vg_test.arrow"
    temp_lxmert_answers: str = "labels/lxmert_answers.json"
    temp_lxmert_train: tuple = "lxmert_data/train/"
    temp_lxmert_eval: str = "lxmert_data/mscoco_minival.json"
    temp_lxmert_test: str = ""

    def __init__(self, **kwargs):
        for f in fields(self):
            str_field = f.name
            if str_field in kwargs:
                setattr(self, str_field, kwargs.pop(str_field))
