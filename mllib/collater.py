import inspect


class CollatedSets:
    def __init__(self, *args):
        self.args = args
        self.range2listpos = {}
        start = 0
        for i, a in enumerate(args):
            self.range2listpos[range(start, len(a) + start)] = i
            start += len(a)

    def __getitem__(self, x):
        if x >= len(self):
            raise IndexError(f"index {x} is out of range 0 to {len(self)}")
        for rng in self.range2listpos:
            if x in rng:
                listpos = self.range2listpos[rng]
                listind = x - rng.start
                return self.args[listpos][listind]

    def __len__(self):
        return sum(map(lambda x: len(x), self.args))

    def __iter__(self):
        return iter(map(lambda x: self[x], range(0, len(self))))


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


def apply_args_to_func(func, kwargs=None):
    func_input = {}
    if kwargs == None:
        kwargs = {}
    req, keyw = get_func_signature_v2(func)
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


def myfunc(x, y, z, a=1, b=2, c=3):
    print(x, y, z, a, b, c)
    return "success"


print(apply_args_to_func(myfunc, {"x": 1, "y": 1, "z": 1, "w": 2, "c": 4}))
