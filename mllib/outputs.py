import pickle as pkl
from typing import Dict, List, Union


class LoopOutputs:
    accuracy: float = None
    lrs: Union[float, List[float]] = None
    losses: Union[float, List[float]] = None

    def __init__(self, losses=0.0, accuracy=0.0, total=None, right=None, lrs=None):
        self.losses = losses
        self.lrs = lrs
        if accuracy is None and (total is not None and right is not None):
            accuracy = f"{right/float(total)*100.0:.3f}"
        elif accuracy is not None:
            accuracy = f"{accuracy:.3f}"
        self.accuracy = accuracy


class ExperimentOutputs:
    epoch_dict: Dict[str, Dict[str, LoopOutputs]] = {}
    name = ''
    datasets = ''
    loops_to_models = {}
    cur_step = 0
    cur_epoch = 0
    epochs = 0
    total_steps = 0
    schedulers = {}

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def add(self, epoch: int, epoch_output: dict, **kwargs):
        for k, v in epoch_output.items():
            assert isinstance(v, LoopOutputs)
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.epoch_dict[f'epoch_{epoch}'] = epoch_output

    def dump(self, fp):
        with open(fp, "wb") as f:
            pkl.dump(self, f)
