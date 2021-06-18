from vltk import OPTIMPATH
from vltk.inspection import import_funcs_from_file


class Optim:
    def __init__(self):
        if "OPTIMDICT" not in globals():
            global OPTIMDICT
            OPTIMDICT = import_funcs_from_file(OPTIMPATH, pkg="vltk.processing")

    def avail(self):
        return list(OPTIMDICT.keys())

    def get(self, name):
        return OPTIMDICT[name]

    def add(self, name, lab):
        OPTIMDICT[name] = lab
