import timeit
from functools import wraps


def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)

        return repl

    return layer


def get_duration(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        starttime = timeit.default_timer()
        output = func(*args, **kwargs)
        print(f"exec: {func.__name__} in {timeit.default_timer() - starttime:.3f} s")
        return output

    return wrapper


def external_config(config_class):
    setattr(config_class, "_identify", None)
    assert hasattr(config_class, "_identify")
    return config_class


@parametrized
def named_model(class_name, name):
    class_name.name = name
    return class_name
