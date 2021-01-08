from mllib import utils
from mllib.maps import dirs

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
