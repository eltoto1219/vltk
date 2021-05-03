import os
from typing import List, Union

from vltk.abc import config
from vltk.inspection import get_args
from vltk.processing.image import Image


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
    objects_file: Union[str, None] = None
    text_processors: Union[None, List[str], str] = ["one_hot_label"]
    image: Union[config.Config] = None
    label_processor: Union[None, str] = "one_hot_label"
    labels: Union[None, str] = None
    eval_datasets = None
    train_datasets = None
    rand_feats: Union[None, tuple] = None
    eval_batch_size = 32
    train_batch_size = 64
    extractor: Union[None, str] = None
    textfile_extensions: Union[List[str], str] = ["json", "jsonl"]
    datadir: str = "/playpen1/home/avmendoz/data"
    img_first: bool = False
    shuffle: bool = True
    num_workers: int = 8
    drop_last: bool = True
    pin_memory: bool = False
    sent_length: int = 20
    max_objects: int = 36
    attribute_file: str = ""
    object_file: str = ""
    img_format: str = "jpg"
    percent: int = 1.0
    skip_eval: bool = False
    skip_train: bool = False
    num_attrs: int = 400
    num_objects: int = 1600
    ignore_id: int = -100
    # word_mask_rate: float = 0.15 # these all belong to text preporcessors. change
    # feature_mask_rate: float = 0.15
    # random_feature_rate: float = 0.10
    # random_word_rate: float = 0.10
    # sentence_match_rate: float = 0.50
    truncate_sentence: bool = True
    return_token_type_ids: bool = True
    add_special_tokens: bool = True
    return_tensors: str = "pt"
    return_attention_mask: bool = True
    use_raw_imgs: bool = False
    pos_dim: int = 4
    visual_dim: int = 2048
    max_detections: str = 36
    annotations: bool = True
    tokenizer = "BertWordPeiceTokenizer"

    def __init__(self, finetune=True, **kwargs):
        super().__init__(**kwargs)
        self.text_processors = Config.handle_iterables(self.text_processors)
        self.textfile_extensions = Config.handle_iterables(self.textfile_extensions)
        self.image = ProcessorConfig(**kwargs.get("image", {}))
        # raise Exception(self.eval_datasets, self.train_datasets)


class ProcessorConfig(config.Config):
    transforms: List = ["ToPILImage", "ToTensor"]

    def __init__(self, **kwargs):
        for f, v in kwargs.items():
            setattr(self, f, v)
            self._overwritten[f] = v

    def build(self):
        from torchvision.transforms import transforms

        _image = Image()
        kwargs = self.to_dict()
        transformlist = kwargs.pop("transforms")
        funcs = []
        for t in transformlist:
            name = t
            t = _image.get(t)
            if name == "Lambda":
                assert hasattr(self, "lambd"), "Lambda transform requires a lambd arg"
                funcs.append(t(self.lambd))
                continue
            args = get_args(t, kwargs)
            if args is None:
                funcs.append(t())
            else:
                funcs.append(t(**args))
        return transforms.Compose(funcs)


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
