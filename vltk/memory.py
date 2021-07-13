# inspiration / code adapted from detectron2

import subprocess

import torch


# from facebook detectron
def handle_cuda_oom(func, *args, kwargs_1, kwargs_2):
    """
    A context which ignores CUDA OOM exception from pytorch.
    """
    try:
        yield func(*args, **kwargs_1)
    except RuntimeError as e:
        # NOTE: the string may change?
        if "CUDA out of memory. " in str(e):
            torch.cuda.empty_cache()
            yield func(*args, **kwargs_2)
        else:
            raise


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
