import functools
import timeit


def get_duration(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        starttime = timeit.default_timer()
        output = func(*args, **kwargs)
        print(f"exec: {func.__name__} in {timeit.default_timer() - starttime:.3f} s")
        return output

    return wrapper
