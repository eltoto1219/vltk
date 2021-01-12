import os

# other utils
from vltk.utils import import_funcs_from_file

BASEPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCHEDPATH = os.path.join(BASEPATH, "processing/Sched.py")
LABELPROCPATH = os.path.join(BASEPATH, "processing/Label.py")
IMAGEPROCPATH = os.path.join(BASEPATH, "processing/Image.py")
OPTIMPATH = os.path.join(BASEPATH, "processing/Optim.py")
FEATUREPATH = os.path.join(BASEPATH, "features.py")
# dictionaries


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


class Label:
    def __init__(self):
        if "LABELPROCDICT" not in globals():
            global LABELPROCDICT
            LABELPROCDICT = import_funcs_from_file(
                LABELPROCPATH, pkg="vltk.processing"
            )

    def avail(self):
        return list(LABELPROCDICT.keys())

    def get(self, name):
        return LABELPROCDICT[name]

    def add(self, name, lab):
        LABELPROCDICT[name] = lab


class Feature:
    def __init__(self):
        if "FEATUREDICT" not in globals():
            global FEATUREDICT
            FEATUREDICT = {}  # import_funcs_from_file(FEATUREPATH, pkg="vltk")

    def avail(self):
        return list(FEATUREDICT.keys())

    def get(self, name):
        return FEATUREDICT[name]

    def add(self, name, feat):
        FEATUREDICT[name] = feat


class Image:
    def __init__(self):
        if "IMAGEPROCDICT" not in globals():
            global IMAGEPROCDICT
            IMAGEPROCDICT = import_funcs_from_file(
                IMAGEPROCPATH, pkg="vltk.processing"
            )

    def avail(self):
        return list(IMAGEPROCDICT.keys())

    def get(self, name):
        return IMAGEPROCDICT[name]

    def add(self, name, proc):
        IMAGEPROCDICT[name] = proc
