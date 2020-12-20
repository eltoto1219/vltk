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
            accuracy = f"{right/float(total)*100.0:.3f}%"
        elif accuracy is not None:
            accuracy = f"{accuracy:.3f}%"
        self.accuracy = accuracy


class ExperimentOutputs:
    epoch_dict: Dict[str, Dict[str, LoopOutputs]] = {}

    def __init__(self, epoch=None, **kwargs):
        if epoch is not None:
            for k, v in kwargs.items():
                assert isinstance(v, LoopOutputs)
            self.epoch_dict[f'epoch_{epoch}'] = kwargs
            setattr(self, f'epoch_{epoch}', kwargs)

    def add(self, epoch, **kwargs):
        for k, v in kwargs.items():
            assert isinstance(v, LoopOutputs)
        self.epoch_dict[f'epoch_{epoch}'] = kwargs
        setattr(self, f'epoch_{epoch}', kwargs)

    def dump(self, fp):
        with open(fp, "wb") as f:
            pkl.dump(self, f)
