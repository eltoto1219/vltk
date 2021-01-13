import os
from typing import List, Union

from vltk.abc import config
from vltk.utils import get_most_free_gpu, get_nvidia_gpu_memory


class LxmertConfig(config.Config):
    from_transformers: bool = True
    ckp_name_or_path = "unc-nlp/lxmert-base-uncased"
    known_labels: int = 1842
    r_layers: int = 5
    l_layers: int = 9
    x_layers: int = 5


class ViTConfig(config.Config):
    from_transformers: bool = False
    vit_variant: Union[str, None] = "ViT-B_16"
    ckp_name_or_path: str = ""


class ModelsConfig(config.Config):
    names = ("lxmert", "vit")
    main_model: str = "lxmert"
    aux_models: tuple = ("frcnn", "vit")
    name_to_config = {"lxmert": LxmertConfig, "vit": ViTConfig}

    def __init__(self, **kwargs):
        new_conf = kwargs.get("name_to_config", False)
        if not new_conf:
            new_conf = kwargs.get("models", {}).get("name_to_config", False)
        if new_conf:
            self.name_to_config = new_conf
        for m in self.names:
            setattr(self, m, self.name_to_config[m]())
        super().__init__(**kwargs)
        self.update(kwargs.get("models", {}))


class PretrainConfig(config.Config):
    epochs: int = 4
    task_matched: bool = False
    task_mask_lm: bool = False
    task_obj_predict: bool = False
    visual_attr_loss: bool = False
    visual_obj_loss: bool = False
    visual_feat_loss: bool = False


class EvalConfig(config.Config):
    half_precision: bool = True
    task_matched: bool = False
    task_mask_lm: bool = False
    task_obj_predict: bool = False
    visual_attr_loss: bool = False
    visual_obj_loss: bool = False
    visual_feat_loss: bool = False


class FinetuneConfig(config.Config):
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


class DataConfig(config.Config):
    pretokenize_procs: List[str] = ["matched_sentence_modeling"]
    processors : Union[None, List[str]] = ["one_hot_label"]
    eval_dataset = "gqa"
    eval_batch_size = 32
    train_batch_size = 64
    label_processor: str = "label_default"
    min_label_occurence = 9
    extractor: Union[None,str] = "frcnn"
    imgid_aliases: set = {"img", "image", "imgid", "img_id", "iid", "image_id"}
    text_aliases: set = {"text", "sent", "que", "question"}
    eval_aliases: set = {"testdev", "eval", "dev", "evaluation", "inference"}
    train_aliases: set = {"train", "finetune", "pretrain"}
    test_aliases: set = {"test"}
    valid_aliases: set = {"val", "valid", "validation"}
    label_aliases: set = {"label", "truth", "answer", "gold"}
    textfile_extensions: Union[List[str], str] = ["json", "jsonl"]
    datadirs: Union[List[str], str] = "/playpen1/home/avmendoz/data"
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
    skip_train: bool = False
    train_split: bool = "trainval"
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
    visual_dim: int = 2048
    max_detections: str = 36
    vit_pretrained_dir = "vit/"
    min_label_frequency: int = 14
    datasets: Union[List[str], str] = ""
    image_preprocessor = "img_to_tensor"


class Config(config.Config):
    data: DataConfig = None
    models: ModelsConfig = None
    eval: EvalConfig = None
    train: Union[FinetuneConfig, PretrainConfig] = None
    logging: bool = True
    gpu: int = None
    aux_gpu: int = None
    seed: int = 9595
    percent_min_gpu_free_mem: float = 0.75
    print_config: bool = True
    datasets: Union[None, str] = None
    base_logdir: str = os.path.join(os.environ.get("HOME", os.getcwd()), "logs")
    rel_logdir: str = ""
    logdir: str = None
    test_save: bool = False
    save_on_crash = False
    save_after_exp = True
    save_after_epoch = False
    email = None
    private_file = None

    def __init__(self, finetune=True, **kwargs):
        super().__init__(**kwargs)
        self.logdir = os.path.join(self.base_logdir, self.rel_logdir)
        if finetune:
            self.train = FinetuneConfig(**kwargs.get("train", {}))
        else:
            self.train = PretrainConfig(**kwargs.get("train", {}))

        self.eval = EvalConfig(**kwargs.get("eval", {}))
        self.data = DataConfig(**kwargs.get("data", {}))
        self.models = ModelsConfig(**kwargs.get("models", {}))

        self._set_gpus()

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
