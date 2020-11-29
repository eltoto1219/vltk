class cheat_config(dict):
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

    def __setattr__(self, k, v):
        self.__dict__[k] = v
