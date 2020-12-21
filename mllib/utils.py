import collections
import contextlib
import functools
import os
import smtplib
import subprocess
import timeit
from collections import defaultdict
from datetime import datetime
from email.message import EmailMessage

import numpy as np
import torch
import yaml


@contextlib.contextmanager
def dummy_context():
    yield None


# with contextlib.suppress(FileNotFoundError):
# os.remove(filename)
# collections.mutablemapping???

def send_email(address, message, failure=True):
    sender = os.environ.get("HOSTNAME", "localhost")
    msg = EmailMessage()
    msg.set_content(message)
    if failure:
        msg['Subject'] = 'MLLIB failure!'
    else:
        msg['Subject'] = 'MLLIB success!'
    msg['From'] = sender
    msg['To'] = [address]
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()


def logfile_name(base):
    specifications = ['year', 'month', 'day', 'hour']
    date = datetime.now()
    date = ':'.join([str(getattr(date, s)) for s in specifications])
    return base + "_" + date


def unflatten_dict(d):
    ret = defaultdict(dict)
    for k, v in d.items():
        k1, delim, k2 = k.partition('.')
        if delim:
            ret[k1].update({k2: v})
        else:
            ret[k1] = v
    return ret


def load_yaml(flags: dict):
    # allow flags to overwrite keys present in yaml file
    if "yaml" in flags:
        yaml_path = flags.pop("yaml")
        yam = yaml.load(open(yaml_path), Loader=yaml.Loader)
        for y in yam:
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


def get_duration(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        starttime = timeit.default_timer()
        output = func(*args, **kwargs)
        print(f"exec: {func.__name__} in {timeit.default_timer() - starttime:.3f} s")
        return output

    return wrapper


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
