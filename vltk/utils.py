import collections
import contextlib
import importlib
import inspect
import json
import os
import smtplib
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from email.message import EmailMessage
from typing import Tuple, Union

import datasets
import jsonlines
import numpy as np
import pyarrow
import torch
import yaml
from torch import nn

PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "libdata"
)


def update_config_with_logdir(config, flags, name, datasets):
    if "base_logdir" not in flags:
        baselogdir = config.base_logdir
    else:
        baselogdir = flags.pop("base_logdir")
    if "rel_logdir" not in flags:
        specifications = ["year", "month", "day", "hour"]
        date = datetime.now()
        date = ":".join([str(getattr(date, s)) for s in specifications])
        rellogdir = f"{name}_" + date
    else:
        rellogdir = flags.pop("rel_logdir")
    if rellogdir == baselogdir:
        rellogdir = ""

    config.update(
        {
            "logdir": os.path.join(baselogdir, rellogdir),
            "rel_logdir": rellogdir,
            "base_logdir": baselogdir,
        }
    )


def get_classes(path_or_dir_name, cls_defintion=None, pkg=None):
    if os.path.isfile(path_or_dir_name):
        clsses = import_classes_from_file(path_or_dir_name, pkg=pkg)
        filter_dict = {}
        if cls_defintion is not None:
            for n, c in clsses.items():
                if cls_defintion in inspect.getmro(c) or hasattr(c, "name"):
                    try:
                        filter_dict[c.name] = c
                    except Exception:
                        filter_dict[n] = c
            return filter_dict
        else:
            return clsses

    classes = {}
    for p in os.listdir(path_or_dir_name):
        if p[0] != "_":
            npkg = pkg + f".{p.split('.')[0]}"
            try:
                mod = importlib.import_module(npkg)
                mod = inspect.getmembers(mod, inspect.isclass)
                for t in mod:

                    if cls_defintion in inspect.getmro(t[-1]):
                        # print(t[-1].__abstractmethods__)
                        if not inspect.isabstract(t[-1]):
                            try:
                                classes[t[-1].name] = t[-1]
                            except AttributeError:
                                classes[t[0]] = t[-1]
            except ModuleNotFoundError:
                pass
    return classes


# from datasets import Dataset
def my_import(name):
    components = name.split(".")
    p = ".".join(components[:-1])
    mod = __import__(p, fromlist=[components[-1]])
    klass = getattr(mod, components[-1])
    return klass


def set_metadata(tbl, tbl_meta={}):
    fields = []
    for f in tbl.schema.names:
        fields.append(tbl.schema.field_by_name(f))

    tbl_metadata = tbl.schema.metadata
    for k, v in tbl_meta.items():
        tbl_metadata[k] = json.dumps(v).encode("utf-8")

    schema = pyarrow.schema(fields, metadata=tbl_metadata)
    tbl = pyarrow.Table.from_arrays(list(tbl.itercolumns()), schema=schema)

    return tbl


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


def batcher(iterable, n=64):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def try_load_json(filepath):
    try:
        with open(filepath) as f:
            yield json.load(f)
    except json.decoder.JSONDecodeError:
        with open(filepath) as f:
            yield jsonlines.open(f)


# def get_classes(path, cls_defintion, pkg):
#     if os.path.isfile(path):
#         modules = import_classes_from_file(path, pkg)
#     else:
#         modules = import_from_dir(path, pkg)
#     filter_mods = [m for m in modules if is_cls(m[1], cls_defintion)]
#     cls_dict = {}
#     for (name, inst) in filter_mods:
#         # inst = importlib.reload(inst)
#         name = getattr(inst, "name", name)
#         cls_dict[name] = inst
#     return cls_dict


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


def clean_imgid(img_id):
    return str(img_id).replace(" ", "")


def load_arrow(dset_to_arrow_fp: dict, fields: Union[Tuple[str], str, None] = None):
    if fields is not None and not fields:
        return None
    arrow_dict = {}
    for dset in dset_to_arrow_fp:
        arrow_fp = dset_to_arrow_fp[dset]
        arrow = datasets.Dataset.from_file(arrow_fp)
        if fields is not None:
            fields = list(fields)
        arrow.set_format(type="numpy", columns=fields)
        arrow_dict[dset] = arrow
    return arrow_dict


def clip_img_ids(img_ids, percent_data=1.0):
    if percent_data != 1.0:
        stop_int = max(1, int(np.ceil(len(img_ids) * percent_data)))
        img_ids = img_ids[:stop_int]
    assert len(img_ids) > 0
    return img_ids


@contextlib.contextmanager
def dummy_context():
    yield None


def send_email(address, message, failure=True):
    sender = os.environ.get("HOSTNAME", "localhost")
    msg = EmailMessage()
    msg.set_content(message)
    if failure:
        msg["Subject"] = "MLLIB failure!"
    else:
        msg["Subject"] = "MLLIB success!"
    msg["From"] = sender
    msg["To"] = [address]
    s = smtplib.SMTP("localhost")
    s.send_message(msg)
    s.quit()


def unflatten_dict(d):
    ret = defaultdict(dict)
    for k, v in d.items():
        k1, delim, k2 = k.partition(".")
        if delim:
            ret[k1][k2] = v
        else:
            ret[k1] = v
    return dict(ret)


def load_yaml(flags: dict):
    # allow flags to overwrite keys present in yaml file
    if "yaml" in flags:
        yaml_path = flags.pop("yaml")
        yam = yaml.load(open(yaml_path), Loader=yaml.Loader)
        for y in yam:
            if isinstance(yam[y], dict):
                if y in flags:
                    for nk, nv in yam[y].items():
                        if nk not in flags[y]:
                            flags[y][nk] = nv
                else:
                    flags[y] = yam[y]
            else:
                if y not in flags:
                    flags[y] = yam[y]
    return flags


def convert_jax_to_torch_weights(
    torch_dict, jax_dict, torch_encoder_weight_name="transformer"
):

    torch_to_jax_alias = [
        ("bias", "bias"),
        ("norm1", "LayerNorm_0"),
        ("norm2", "LayerNorm_2"),
        ("norm", "encoder_norm"),
        ("weight", "kernel"),
        ("weight", "scale"),
        ("transformer", "Transformer"),
        ("encoder_layers.", "encoderblock_"),
        ("attn", "MultiHeadDotProductAttention_1"),
        ("out", "out"),
        ("query", "query"),
        ("key", "key"),
        ("value", "value"),
        ("mlp", "MlpBlock_3"),
        ("fc1", "Dense_0"),
        ("fc2", "Dense_1"),
        ("pos_embedding", "posembed_input"),
        ("cls_token", "cls"),
        ("classifier", "head"),
    ]
    # pos embdedding = n_patches +  1 (for cls token)

    jax_flattened_weights = flatten_dict(jax_dict, parent_key="")
    jax_dict_renamed = collections.OrderedDict()
    for k, v in jax_flattened_weights.items():
        for (tn, jn) in torch_to_jax_alias:
            k = k.replace(jn, tn)
        jax_dict_renamed[k] = torch.tensor(v.tolist())  # .tolist()
    for j, t in zip(sorted(jax_dict_renamed), sorted(torch_dict)):
        if j != t:
            print(j, t)
        if jax_dict_renamed[j].shape != torch_dict[t].shape:
            jshape = list(jax_dict_renamed[j].shape)
            tshape = list(torch_dict[t].shape)
            assert len(jshape) == len(tshape)
            if sum(jshape) == sum(tshape):
                ind_map = [0] * len(jshape)
                seen_inds = set()
                added_inds = set()
                for ji, jv in enumerate(jshape):
                    for ti, tv in enumerate(tshape):
                        if jv == tv:
                            if ji not in seen_inds and ti not in added_inds:
                                ind_map[ti] = ji
                                seen_inds.add(ji)
                                added_inds.add(ti)
                try:
                    new_val = jax_dict_renamed[j].permute(*tuple(ind_map))
                except Exception:
                    raise Exception(
                        ind_map, jax_dict_renamed[j].shape, torch_dict[j].shape, j
                    )
                assert new_val.shape == torch_dict[t].shape, (
                    new_val.shape,
                    torch_dict[t].shape,
                    ind_map,
                    jshape,
                )
                jax_dict_renamed[j] = new_val
            else:
                print(f"SKIPPIG: mismatched {j, t}, shapes {jshape, tshape}")
                jax_dict_renamed[j] = torch_dict[t]
    assert len([x for x in jax_dict_renamed if torch_encoder_weight_name in x]) == len(
        [x for x in torch_dict if torch_encoder_weight_name in x]
    )
    return jax_dict_renamed


def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return collections.OrderedDict(items)


# tensor equality
def tensor_equality(a, b):

    n1 = a.numpy()
    n2 = b.numpy()
    print(n1.shape)
    print(n2.shape)
    print(n1[0, 0, :5])
    print(n2[0, 0, :5])
    assert np.allclose(
        n1, n2, rtol=0.01, atol=0.1
    ), f"{sum([1 for x in np.isclose(n1, n2, rtol=0.01, atol=0.1).flatten() if x == False])/len(n1.flatten())*100:.4f} % element-wise mismatch"
    raise Exception("tensors are all good")


def get_subfiles_from_path(path_or_dir, relative=None) -> list:
    file_list = set()
    if isinstance(path_or_dir, list):
        if relative is None:
            x = [os.path.abspath(i) for i in path_or_dir]
        else:
            x = [os.path.join(relative, i) for i in path_or_dir]
    else:
        if relative is None:
            x = [os.path.abspath(path_or_dir)]
        else:
            x = [os.path.join(relative, path_or_dir)]
    for fp in x:
        if os.path.isdir(fp):
            for path, subdirs, files in os.walk(fp):
                for name in files:
                    file_list.add(os.path.join(path, name))
        elif os.path.isfile(fp):
            file_list.add(fp)
        else:
            raise Exception(fp)
    return list(file_list)


def get_most_free_gpu():
    if not torch.cuda.is_available():
        return -1
    mem_list = get_nvidia_gpu_memory()
    return min(
        list(map(lambda k: (k, mem_list[k][0] / mem_list[k][1]), mem_list)),
        key=lambda x: x[1],
    )[0]


def get_nvidia_gpu_memory():
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,nounits,noheader",
        ],
        encoding="utf-8",
    )
    gpu_memory = [eval(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map
