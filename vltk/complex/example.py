import torch
import vltk.dataset as data
from vltk import dataset
from vltk.abc.experiment import Experiment
from vltk.abc.loop import Loop


# define loop
class DataLoop(Loop):
    name: str = "data"
    is_train: bool = False

    def forward(self, batch):
        return super().forward(batch)

    def loop(self, batch, model_outputs):
        return super().loop(batch, model_outputs)


# now define epxeirment
class Data(Experiment):
    name: str = "data"
    loops_to_models: dict = {DataLoop: [None], DataLoop.eval_instance(): [None]}

    def loginfo(self):
        return super().loginfo()

    def keys(self):
        entry = None
        for loop_name, loop in self:
            for x in loop:
                _ = x.pop("image", None)
                _ = x.pop("roi_features", None)
                entry = x
                # raise Exception(x)
                for k, v in entry.items():
                    if isinstance(v, torch.Tensor):
                        pass
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
                data.UniversalDataset.transpose_img2txt(
                    entry, img_keys=["raw_imgs", "raw_sizes"], device="cpu"
                )
                for k, v in entry.items():
                    shape = None
                    if isinstance(v, torch.Tensor):
                        shape = v.shape
                    print(k, type(v), shape)
                break
