from vltk import compat, 
from vltk.abc.complex import ComplexExperiments
from vltk.abc.visndatasetadapter import VisnDatasetAdapters
from vltk.abc.simple import SimpleExperiments
from vltk.configs import Config

_complex_experiments = ComplexExperiments()
_simple_experiments = SimpleExperiments()


def run_experiment(config, flags, name_or_exp, datasets):
    global experiment
    experiment = ""
    if config.print_config:
        print(config)
    if isinstance(name_or_exp, str):
        .update_config_with_logdir(config, flags, name_or_exp, datasets)
        experiment = _complex_experiments.get(name_or_exp)(
            config=config, datasets=datasets
        )
        experiment()
    else:
        .update_config_with_logdir(config, flags, name_or_exp.name, datasets)
        experiment = name_or_exp(config=config, datasets=datasets)
        experiment()


def run_simple_experiment(config, flags, name_or_exp, datasets):

    global experiment
    experiment = ""
    if config.print_config:
        print(config)
    if isinstance(name_or_exp, str):
        .update_config_with_logdir(config, flags, name_or_exp, datasets)
        experiment = _simple_experiments.get(name_or_exp)(
            config=config, datasets=datasets
        )
        experiment()
    else:
        .update_config_with_logdir(config, flags, name_or_exp.name, datasets)
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
    _visndatasetadapters = VisnDatasetAdapters()
    # _models = dirs.Models()
    # will need to fix
    VisnDatasetAdapter = _visndatasetadapters.get(extractor)
    # Model = _models.get(extractor)
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
        # model = Model.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnnconfig)
    else:
        model = None

    gpu = config.gpu
    config = config.data

    VisnDatasetAdapter.extract(
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
