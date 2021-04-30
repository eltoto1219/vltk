from vltk import SCHEDPATH
from vltk.inspection import import_funcs_from_file


class Sched:
    def __init__(self):
        if "SCHEDDICT" not in globals():
            global SCHEDDICT
            SCHEDDICT = import_funcs_from_file(SCHEDPATH, pkg="vltk.processing")

    def avail(self):
        return list(SCHEDDICT.keys())

    def get(self, name):
        return SCHEDDICT[name]

    def add(self, name, lab):
        SCHEDDICT[name] = lab
