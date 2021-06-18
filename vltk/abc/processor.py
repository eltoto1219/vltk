import torch
from vltk.inspection import collect_args_to_func


class Processor:
    _type = None
    _keys = ()

    @property
    def keys(self):
        return self._keys

    def __init__(self, *args):
        for a in args:
            assert (
                a == str
            ), f"{a} must be key name to be queried if input to forward \
                    will be a dictionaty"
            self._keys += (a,)

    @torch.no_grad()
    def __call__(self, inp, *args, **kwargs):
        if isinstance(inp, dict):
            assert (
                self.keys
            ), "the `keys` attribute, set as optional positional arguments to \
                    the __init__ method must be set when the input to the forward method \
                    is a dictionary"
        kwargs = collect_args_to_func(self.forward, kwargs)
        output = self.forward(inp, *args, **kwargs)
        if not isinstance(output, dict):
            assert isinstance(
                output, torch.Tensor
            ), "the outputs of any processor must be a torch tensor or a \
            dictionary where the repective value(s) from the key(s) of interest, specified in the init method, \
            must be a torch tensor aswell"
        else:
            pass
            # assert not any(
            #     map(
            #         lambda x: isinstance(output[x], torch.Tensor)
            #         if x in output
            #         else True,
            #         self.keys,
            #     )
            # ), f"Not all values to the respective keys, {self.keys}, are torch tensors"
        return output


class VisnProccessor(Processor):
    _type = "visn"


class LangProccessor(Processor):
    _type = "lang"


class VisnLangProccessor(Processor):
    _type = "visnlang"
