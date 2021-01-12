from vltk import compat, utils
from vltk.configs import Config
from vltk.maps import dirs

_experiments = dirs.Exps()


def run_experiment(config, flags, name_or_exp, datasets):
    if config.print_config:
        print(config)
    if isinstance(name_or_exp, str):
        utils.update_config_with_logdir(config, flags, name_or_exp, datasets)
        exp_from_str = _experiments.get(name_or_exp)(config=config, datasets=datasets)
        exp_from_str()
    else:
        utils.update_config_with_logdir(config, flags, name_or_exp.name, datasets)
        global experiment
        experiment = name_or_exp(config=config, datasets=datasets)
        experiment()


def extract_data(
    extractor,
    dataset,
    config=None,
    splits=None,
    features=None,
    image_preprocessor=None,
    img_format=None,
    flags=None,
):
    if config is None:
        config = Config(**flags)
    if flags is None:
        flags = {}
    _imagesets = dirs.Imagesets()
    _models = dirs.Models()
    Imageset = _imagesets.get(extractor)
    Model = _models.get(extractor)
    if "features" in flags:
        features = flags.pop("features")
    if splits is None:
        splits = flags.pop("splits", None)
    if "image_preprocessor" in flags:
        image_preprocessor = flags.get("image_preprocessor")
    if img_format is None:
        img_format = flags.get("img_format")
    else:
        img_format = "jpg"
    # hard code for now:
    if extractor == "frcnn":
        frcnnconfig = compat.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        frcnnconfig.model.device = config.gpu
        model = Model.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnnconfig)
    else:
        model = None

    gpu = config.gpu
    config = config.data

    Imageset.extract(
        dataset_name=dataset,
        config=config,
        model=model,
        image_preprocessor=image_preprocessor,
        features=features,
        splits=splits,
        device=gpu,
        max_detections=config.max_detections,
        pos_dim=config.pos_dim,
        visual_dim=config.visual_dim,
    )
