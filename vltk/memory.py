# inspiration / code adapted from detectron2

import torch


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
