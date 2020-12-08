import collections
import functools
import os
import subprocess
import timeit

import numpy as np


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


# from google
def _flatten_dict(d, parent_key="", sep="/"):
    """Flattens a dictionary, keeping empty leaves."""
    items = []
    for k, v in d.items():
        path = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(_flatten_dict(v, path, sep=sep).items())
        else:
            items.append((path, v))

    # Keeps the empty dict if it was set explicitly.
    if parent_key and not d:
        items.append((parent_key, {}))

    return dict(items)


def get_duration(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        starttime = timeit.default_timer()
        output = func(*args, **kwargs)
        print(f"exec: {func.__name__} in {timeit.default_timer() - starttime:.3f} s")
        return output

    return wrapper


def get_file_path(path_or_dir, relative=None):
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
