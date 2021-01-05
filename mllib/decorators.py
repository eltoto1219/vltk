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


def external_config(config_class):
    @functools.wraps(config_class)
    def wrapper():
        setattr(config_class, "_identify", None)
        assert hasattr(config_class, "_identify", None)
        return config_class

    return wrapper
