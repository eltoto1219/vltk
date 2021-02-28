import os
from typing import List, Union

from vltk.abc import config
from vltk.utils import get_most_free_gpu, get_nvidia_gpu_memory


class ViTConfig(config.Config):
    from_transformers: bool = False
    vit_variant: Union[str, None] = "ViT-B_16"
    ckp_name_or_path: str = ""


class ModelConfig(config.Config):
    checkpoint = None
    freeze_layers = None
    freeze_embeddigs = None
    freeze_heads = None

    def __init__(self, **kwargs):
        for f, v in kwargs.items():
            setattr(self, f, v)
            self._overwritten[f] = v


class ModelsConfig(config.Config):
    main_model: str = "lxmert"
    checkpoint = None
    all_on_same_device = False
    models_to_devices = None

    def add(self, model_name, model_config):
        model_base = model_name.split("_")[0]
        attr_dict = {}
        if hasattr(self, model_base):
            for attr, attr_val in getattr(self, model_base).items():
                attr_dict[attr] = attr_val
            mconf = model_config(**attr_dict)

            setattr(self, model_base, mconf)
        else:
            raise Exception

    def __init__(self, **kwargs):
        for f, v in kwargs.items():

            if isinstance(v, dict):
                v = ModelsConfig(**v)
            setattr(self, f, v)
            self._overwritten[f] = v


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
    eval_aliases: set = {"testdev", "eval", "dev", "evaluation", "inference"}
    train_aliases: set = {"train", "finetune", "pretrain"}
    test_aliases: set = {"test"}
    valid_aliases: set = {"val", "valid", "validation"}
    text_processors: Union[None, List[str], str] = ["one_hot_label"]
    image_processors: Union[None, List[str], str] = []
    label_processor: Union[None, str] = "one_hot_label"
    label_preprocessor: str = "label_default"
    labels: Union[None, str] = None
    eval_datasets = ("gqa", "dev")
    train_datasets = [("gqa", ("train", "val"))]
    rand_feats: Union[None, tuple] = None
    eval_batch_size = 32
    train_batch_size = 64
    min_label_frequency = 9
    extractor: Union[None, str] = "frcnn"
    textfile_extensions: Union[List[str], str] = ["json", "jsonl"]
    datadirs: Union[List[str], str] = "/playpen1/home/avmendoz/data"
    img_first: bool = False
    cache_batch: str = "batch.temp"
    overwrite_cache_batch: bool = False
    shuffle: bool = True
    num_workers: int = 8
    drop_last: bool = True
    pin_memory: bool = False
    sent_length: int = 20
    max_objects: int = 36
    attribute_file: str = ""
    object_file: str = ""
    img_format: str = "jpg"
    percent_data: int = 1.0
    skip_eval: bool = False
    skip_train: bool = False
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
    use_raw_imgs: bool = False
    pos_dim: int = 4
    visual_dim: int = 2048
    max_detections: str = 36
    vit_pretrained_dir = "vit/"
    annotations: bool = True

    # for image processing stuff
    size = 768
    mode = "bicubic"
    scale = "standard"
    gpu = None
    pad_value = 0
    std = None
    mean = None
    inplace = True
    resize = True
    normalize = True
    aspect_ratio = True
    pad = True

    def __init__(self, finetune=True, **kwargs):
        super().__init__(**kwargs)
        self.eval_datasets = Config.handle_iterables(self.eval_datasets)
        if len(self.eval_datasets) > 1 and isinstance(self.eval_datasets[1], str):
            self.eval_datasets = [(self.eval_datasets[0], set([self.eval_datasets[1]]))]
        self.text_processors = Config.handle_iterables(self.text_processors)
        self.image_processors = Config.handle_iterables(self.image_processors)
        self.datadirs = Config.handle_iterables(self.datadirs)
        self.textfile_extensions = Config.handle_iterables(self.textfile_extensions)


class Config(config.Config):
    data: DataConfig = None
    models: ModelsConfig = None
    eval: EvalConfig = None
    train: Union[FinetuneConfig, PretrainConfig] = None
    logging: bool = True
    gpu: int = None
    seed: int = 9595
    percent_min_gpu_free_mem: float = 0.75
    print_config: bool = False
    datasets: Union[None, str] = None
    base_logdir: str = os.path.join(os.environ.get("HOME", os.getcwd()), "logs")
    rel_logdir: str = ""
    logdir: str = (
        None  # this will be determined lated by datadir + base_logdir + rel_logdir
    )
    test_save: bool = False
    save_on_crash = False
    save_after_exp = True
    save_after_epoch = False
    email = None
    experimentdir: Union[None, str] = os.getcwd()
    test_run: bool = True
    break_loop_on_test: bool = True
    empty_cache: bool = True
    launch_blocking: bool = True
    vltk_checkpoint_dir: Union[str, None] = None

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

        if self.launch_blocking:
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        """
        ? a better way to propogate these to subconfigs ?
        """
        setattr(self.data, "test_run", self.test_run)
        setattr(self.data, "logdir", self.logdir)

    # should add more testcases incase all gpus are busy
    def _set_gpus(self):
        if self.gpu is not None:
            pass
        if self.gpu is None:
            self.gpu = get_most_free_gpu()

        if self.gpu == -1:
            print("WARNING: setting everything to cpu")
            self.gpu = "cpu"
