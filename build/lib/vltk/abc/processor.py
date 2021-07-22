import torch
from vltk.inspection import collect_args_to_func


class Processor:
    _type = None
    _keys = ()

    @property
    def keys(self):
        if isinstance(self._keys, str):
            return set([self._keys])
        return set(self._keys)

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @torch.no_grad()
    def __call__(self, inp, **kwargs):
        if isinstance(inp, dict):
            proc_keys = self.keys
            intersection = proc_keys.intersection(set(inp.keys()))
            assert (
                intersection == proc_keys
            ), f"{type(self).__name__} requires {proc_keys} to be present within the input dictionary, but not all \
                    keys are present. the input dictionary only has: {inp.keys()}"

        kwargs = collect_args_to_func(self.forward, kwargs)
        output = self.forward(inp, **kwargs)
        if not isinstance(output, dict):
            assert isinstance(
                output, torch.Tensor
            ), "the outputs of any processor must be a torch tensor or a \
            dictionary where the repective value(s) from the key(s) of interest, specified in the init method, \
            must be a torch tensor aswell"
        else:
            pass

        return output


class VisnProcessor(Processor):
    _type = "visn"


class LangProcessor(Processor):
    _type = "lang"


class VisnLangProcessor(Processor):
    _type = "visnlang"

    @torch.no_grad()
    def __call__(self, text_inp, visn_inp, **kwargs):

        text_inp, visn_inp = self.forward(text_inp, visn_inp, **kwargs)

        return text_inp, visn_inp
