import importlib
import inspect
import os
import sys

import torch.nn as nn

PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "libdata"
)


def get_args(func, dictionary):
    argspec = inspect.getfullargspec(func)
    args = argspec.args
    kwargs = 0 if argspec.varkw is None else 1
    if kwargs:
        return dictionary
    elif args:
        argdict = {}
        for a in args:
            if a in dictionary:
                argdict[a] = dictionary[a]
        return argdict
    else:
        return None


def get_classes(path_or_dir_name, cls_defintion=None, pkg=None):
    os.chdir(os.getcwd())
    # for single file
    if os.path.isfile(path_or_dir_name):
        clsses = import_classes_from_file(path_or_dir_name, pkg=pkg)
        filter_dict = {}
        if cls_defintion is not None:
            for n, c in clsses.items():
                if cls_defintion in inspect.getmro(c) or hasattr(c, "name"):
                    try:
                        filter_dict[c.name] = c
                    except Exception:
                        filter_dict[c.__name__.lower()] = c
            return filter_dict
        else:
            return clsses

    # for dir
    classes = {}

    sys.path.insert(0, path_or_dir_name)
    module_name = None
    for p in os.listdir(path_or_dir_name):
        if p[0] != "_":
            if pkg is None and p[-3:] == ".py":
                module_name = p[:-3]
                npkg = __import__(module_name, fromlist=[""])
            elif pkg is None and p[-3:] != ".py":
                continue
            else:
                npkg = pkg + f".{p.split('.')[0]}"
            try:
                if pkg is None:
                    mod = inspect.getmembers(npkg, inspect.isclass)
                    # if module_name == 'exp_newdata':
                    #    raise Exception(mod)
                else:
                    try:
                        mod = importlib.import_module(npkg)
                    except Exception:
                        mod = importlib.import_module(npkg)
                        # raise Exception(p, npkg, mod)
                    mod = inspect.getmembers(mod, inspect.isclass)
                for t in mod:
                    if cls_defintion in inspect.getmro(t[-1]):
                        # print(t[-1].__abstractmethods__)
                        if not inspect.isabstract(t[-1]):
                            try:
                                classes[t[-1].__name__.lower()] = t[-1]
                            except AttributeError:
                                classes[t[-1].__name__.lower()] = t[-1]
            except Exception:
                mod = importlib.import_module(npkg)
                assert module_name != "exp_newdata"
                pass
    return classes


def get_func_signature_v2(func):
    required = set()
    keyword = {}
    sig = inspect.signature(func).parameters
    for k, v in sig.items():
        if v.default == inspect._empty:
            required.add(k)
        else:
            keyword[k] = v.default
    return required, keyword


def collect_args_to_func(func, kwargs=None, mandatory=False):
    func_input = {}
    if kwargs is None:
        kwargs = {}
    else:
        assert isinstance(kwargs, dict)
    req, keyw = get_func_signature_v2(func)
    if mandatory:
        for r in req:
            assert r in kwargs, (
                "\n"
                f"The required args of {func.__name__} are: {req}"
                f" but '{r}' not found in kwargs: {list(kwargs.keys())}"
            )
            func_input[r] = kwargs[r]
    for k in keyw:
        if k in kwargs:
            func_input[k] = kwargs[k]
    return func_input


def apply_args_to_func(func, kwargs=None, mandatory=True):
    func_input = {}
    if kwargs is None:
        kwargs = {}
    else:
        assert isinstance(kwargs, dict)
    req, keyw = get_func_signature_v2(func)
    if mandatory:
        for r in req:
            assert r in kwargs, (
                "\n"
                f"The required args of {func.__name__} are: {req}"
                f" but '{r}' not found in kwargs: {list(kwargs.keys())}"
            )
            func_input[r] = kwargs[r]
    for k in keyw:
        if k in kwargs:
            func_input[k] = kwargs[k]
    return func(**func_input)


def is_cls(inspect_class, cls_defintion):
    if (
        cls_defintion in inspect.getmro(inspect_class)
        and inspect_class.__name__ != cls_defintion.__name__
    ):
        return True
    else:
        return False


def is_model(inspect_class):
    if (
        nn.Module in inspect.getmro(inspect_class)
        and inspect_class.__name__ != "Module"
    ):
        return True
    else:
        return False


def import_from_dir(clsdir, pkg):
    modules = []
    for p in os.listdir(clsdir):
        file = os.path.join(clsdir, p)
        mods = import_classes_from_file(file, pkg)
        modules.extend(mods)

    return modules


def import_classes_from_file(clspath, pkg=None):
    clsfile = clspath.split("/")[-1]
    clsname = clsfile.split(".")[0]
    if pkg is not None:
        clsname = pkg + f".{clsname}"
    clsdir = clspath.replace(clsfile, "")
    sys.path.insert(0, clsdir)
    mod = importlib.import_module(clsname, package=None)
    return {clss[0]: clss[1] for clss in inspect.getmembers(mod, inspect.isclass)}


def import_funcs_from_file(clspath, pkg):
    clsfile = clspath.split("/")[-1]
    clsname = clsfile.split(".")[0]
    if pkg is not None:
        clsname = pkg + f".{clsname}"
    clsdir = clspath.replace(clsfile, "")
    sys.path.insert(0, clsdir)
    mod = importlib.import_module(clsname, package=pkg)
    return {func[0]: func[1] for func in inspect.getmembers(mod, inspect.isfunction)}


def get_func_signature(func):
    sig = inspect.signature(func).parameters
    return sig
