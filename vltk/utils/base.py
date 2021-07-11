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
from collections.abc import Iterable
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


# merge dictionaries
def mergedicts(dict1, dict2):
    for k in set(dict1.keys()).union(dict2.keys()):
        if k in dict1 and k in dict2:
            if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                yield (k, dict(mergedicts(dict1[k], dict2[k])))
            else:
                # If one of the values is not a dict, you can't continue merging it.
                # Value from second dict overrides one in first and we move on.
                yield (k, dict2[k])
                # Alternatively, replace this with exception raiser to alert you of value conflicts
        elif k in dict1:
            yield (k, dict1[k])
        else:
            yield (k, dict2[k])


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


# this is to update metadata on dataset objects
def set_metadata(tbl, tbl_meta={}):
    fields = []
    for i, f in enumerate(tbl.schema.names):
        fields.append(tbl.schema[i])

    tbl_metadata = tbl.schema.metadata
    for k, v in tbl_meta.items():
        if isinstance(v, dict):
            tbl_metadata[k] = json.dumps(v).encode("utf-8")
        elif isinstance(v, set):
            tbl_metadata[k] = "\n".join(v).encode("utf-8")
        else:
            tbl_metadata[k] = str(v).encode("utf-8")

    schema = pyarrow.schema(fields, metadata=tbl_metadata)
    tbl = pyarrow.Table.from_arrays(list(tbl.itercolumns()), schema=schema)

    return tbl


def batcher(iterable, n=64):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def try_load(filepath):
    ext = str(filepath).split(".")[-1]
    if ext in ("json", "jsonl"):
        try:
            with open(filepath) as f:
                data = json.load(f)
            return data
        except json.decoder.JSONDecodeError:
            with open(filepath) as f:
                data = jsonlines.open(f)
            return data
    elif "pdf" == ext:
        return str(filepath)
    raise Exception(ext, filepath)
    # try:
    #     with open(filepath) as f:
    #         entry = jsonlines.open(f)
    #         yield entry
    # except Exception:
    #     with open(filepath) as f:
    #         data = json.load(f)
    #     return data


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


def unflatten_dict(dictionary):
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict


def load_yaml(flags: dict):
    # allow flags to overwrite keys present in yaml file
    if "yaml" in flags:
        yaml_path = flags.pop("yaml")
        loaded = yaml.load(open(yaml_path), Loader=yaml.Loader)
        updated_flags = dict(mergedicts(loaded, flags))
        return updated_flags
    else:
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


def isprimitive(obj):
    if (
        isinstance(obj, int)
        or isinstance(obj, bool)
        or isinstance(obj, str)
        or isinstance(obj, float)
    ):
        return True
    else:
        return False


def on_children(obj: object, findtype=int, func=None):
    # if object is dict
    if isinstance(obj, dict) and not isinstance(obj, findtype):
        for k, v in obj.items():
            new_v = on_children(v, findtype=findtype, func=func)
            if new_v is not None:
                obj[k] = new_v
    # if the object is a primitive
    elif isprimitive(obj) and not isinstance(obj, findtype):
        return None
    # if the object is iterable
    elif isinstance(obj, Iterable) and not isinstance(obj, findtype):
        for idx, v in enumerate(obj):
            new_v = on_children(v, findtype=findtype, func=func)
            if new_v is not None:
                obj[idx] = new_v
    # if the object is not a findtype, iterable, dictionary, or prim
    elif not isinstance(obj, findtype) and hasattr(obj, "__dict__"):
        raise Exception
    # if the object is the find type
    elif func is not None:
        out = func(obj)
        if out is not None:
            obj = out
        else:
            obj = None
    return obj


def change_device(batch, device="cpu"):
    assert isinstance(device, int) or device == "cpu"
    device = torch.device(device)
    return on_children(batch, findtype=torch.Tensor, func=lambda x: x.to(device))


def check_device(batch):
    return on_children(
        batch,
        findtype=torch.Tensor,
        func=lambda x: (print(x.shape, x.device) if x is not None else x),
    )


def get_list_primitive(ls):

    if isinstance(ls, collections.Iterable) and not isinstance(ls, str):
        return get_list_primitive(ls[0])
    else:
        if ls is None:
            return None
        else:
            return type(ls)


# stack overflow credit
def flatten_stringlist(container):
    if container is None:
        return []
    if isinstance(container, str):
        return [container]
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten_stringlist(i):
                yield j
        else:
            yield i


def get_arrow_primitive(schema_value):
    if hasattr(schema_value, "feature"):
        return get_arrow_primitive(schema_value.feature)
    else:
        return schema_value.dtype


def convertids_recursive(ls, objids):
    ls = list(ls)
    # get deepest nested list
    if (
        isinstance(ls, collections.Iterable)
        and not isinstance(ls, str)
        and isinstance(ls[0], str)
    ):
        ten = torch.Tensor(list(map(lambda x: objids[x], ls)))
        return ten
    else:
        for idx, item in enumerate(ls):
            res = convertids_recursive(item, objids)
            assert isinstance(res, torch.Tensor), (type(res), res)
            ls[idx] = res
        try:
            ls = torch.stack(ls)
        except Exception:
            pass
        return ls
