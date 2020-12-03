import functools
import os
import subprocess
import timeit


def get_duration(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        starttime = timeit.default_timer()
        output = func(*args, **kwargs)
        print(f"exec: {func.__name__} in {timeit.default_timer() - starttime:.3f} s")
        return output

    return wrapper


def get_file_path(data_dir, relative_path):
    files = []
    fp = os.path.join(data_dir, relative_path)
    if os.path.isdir(fp):
        for x in os.listdir(fp):
            files.append(os.path.join(fp, x))
    elif os.path.isfile(fp):
        files.append(fp)
    else:
        raise Exception(fp)

    return files


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


class cheat_config(dict):
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

    def __setattr__(self, k, v):
        pass
