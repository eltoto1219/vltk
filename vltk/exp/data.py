import torch
from vltk import Dataset
from vltk.abc.experiment import Experiment


class Data(Experiment):
    name: str = "data"
    loops_to_models: dict = {"data": [None]}

    def loginfo(self):
        return super().loginfo()

    def keys(self):
        entry = None
        for loop_name, loop in self:
            for x in loop:
                entry = x
                for k, v in entry.items():
                    shape = None
                    if isinstance(v, torch.Tensor):
                        shape = v.shape
                    print(k, type(v), shape)
                break
            break

    def transpose(self):
        assert self.config.data.img_first
        assert not self.config.data.arrow_fields
        for loop_name, loop in self:
            for x in loop:
                entry = x
                for k, v in entry.items():
                    shape = None
                    if isinstance(v, torch.Tensor):
                        shape = v.shape
                    print(k, type(v), shape)
                data.UniversalDataset.transpose_img2txt(entry, img_keys=["raw_imgs", "raw_sizes"], device="cpu")
                for k, v in entry.items():
                    shape = None
                    if isinstance(v, torch.Tensor):
                        shape = v.shape
                    print(k, type(v), shape)
                break


