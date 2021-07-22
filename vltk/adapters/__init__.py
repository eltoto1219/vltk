import inspect
import os
import sys

import vltk.vars as vltk
from vltk.abc.adapter import Adapter
from vltk.abc.extraction import VisnExtraction
from vltk.abc.visnadapter import VisnDataset
from vltk.abc.visnlangadatper import VisnLangDataset
from vltk.inspection import get_classes


class Adapters:
    def __init__(self):
        if "ADAPTERDICT" not in globals():
            global ADAPTERDICT
            ADAPTERDICT = get_classes(vltk.ADAPTERS, Adapter, pkg="vltk.adapters")
        # top = inspect.stack()[-1][1]
        # name = "/".join(top.split("/")[:-1])
        # top = top.split("/")[-1].split(".")[0]
        # sys.path.append(name)
        # __import__(top)
        # clsmembers = inspect.getmembers(sys.modules[top])
        # raise Exception("woo", clsmembers)

    def is_visnlang(self, adapter: str):
        assert adapter in self.avail(), f"adapter {adapter} not is not available"
        adapter_class = self.get(adapter)
        return VisnLangDataset in adapter_class.__bases__

    def is_visn(self, adapter: str):
        assert adapter in self.avail(), f"adapter {adapter} not is not available"
        adapter_class = self.get(adapter)
        return VisnDataset in adapter_class.__bases__

    def is_extraction(self, adapter: str):
        assert adapter in self.avail(), f"adapter {adapter} not is not available"
        adapter_class = self.get(adapter)
        return VisnExtraction in adapter_class.__bases__

    @staticmethod
    def avail():
        return list(ADAPTERDICT.keys())

    def get(self, name):
        try:
            return ADAPTERDICT[name]
        except KeyError:
            raise Exception(f"{name} not available from {self.avail()}")

    def add(self, *args):
        for dset in args:
            ADAPTERDICT[dset.__name__.lower()] = dset
