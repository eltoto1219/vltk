import vltk
from vltk.abc.processor import (LangProccessor, Processor, VisnLangProccessor,
                                VisnProccessor)
from vltk.inspection import get_classes


class Processors:
    def __init__(self):
        if "PROCESSORDICT" not in globals():
            global PROCESSORDICT
            PROCESSORDICT = get_classes(vltk.PROCESSORS, Processor, pkg="vltk.adapters")

    def is_visnlang(self, processor: str):
        assert processor in self.avail(), f"adapter {processor} not is not available"
        processor_class = self.get(processor)
        return processor_class.__bases__[0] == VisnLangProccessor

    def is_visn(self, processor: str):
        assert processor in self.avail(), f"adapter {processor} not is not available"
        processor_class = self.get(processor)
        return processor_class.__bases__[0] == VisnProccessor

    def is_lang(self, processor: str):
        assert processor in self.avail(), f"adapter {processor} not is not available"
        processor_class = self.get(processor)
        return processor_class.__bases__[0] == LangProccessor

    @staticmethod
    def avail():
        return list(PROCESSORDICT.keys())

    def get(self, name):
        try:
            return PROCESSORDICT[name]
        except KeyError:
            raise Exception(f"{name} not available from {self.avail()}")

    def add(self, *args):
        for dset in args:
            PROCESSORDICT[dset.__name__.lower()] = dset
