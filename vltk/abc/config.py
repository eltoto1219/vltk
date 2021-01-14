import json
from collections import Iterable
from typing import Union

import yaml

DELIM = ","


class Config:
    _identify = None
    _overwritten = {}

    @staticmethod
    def handle_iterables(x):
            if not x:
                return x
            # if only one
            if isinstance(x, str):
                return [x]
            # for tuples not nested tuples
            elif isinstance(x, tuple) and isinstance(x[0], str):
                return list(x)
            # for nested tuples or lists
            elif (
                (isinstance(x, list) or isinstance(x, tuple))
                and (isinstance(x[0], list) or isinstance(x, tuple))
            ):
                temp = []
                for t in x:
                    dset = t[0]
                    splt = t[1]
                    if isinstance(splt, str):
                        splt = set([splt])
                    elif not isinstance(splt, set):
                        splt = set(splt)
                    temp.append((dset, splt))
                return temp
            # for not nested lists
            elif isinstance(x, list) and isinstance(x[0], str):
                return x
            else:
                raise Exception(f"input: {x} not accepted")


    def __init__(self, **kwargs):
        for f, v in self:
            if f in kwargs:
                kv = kwargs.get(f)
                if v != kv:
                    setattr(self, f, kv)
                    self._overwritten[f] = v

    def __iter__(self):
        for k in set(self.__class__.__dict__.keys()).union(set(self.__dict__.keys())):
            if k[0] != "_" and not callable(getattr(self, k)):
                yield k, getattr(self, k)

    def __str__(self):
        # or, i could have simply done a yaml dumps
        string = ""
        for k, v in self:
            if hasattr(v, "_identify"):
                string += f"{k}:\n"
                string += "".join([f"--{vsub}\n" for vsub in str(v).split("\n")])
            else:
                string += f"{k}:{v}\n"
        return string[:-1]

    @staticmethod
    def parse(arg):
        if isinstance(arg, str) and DELIM in arg:
            arg = arg.split(DELIM)
            if len(arg) == 0:
                arg = ""
            else:
                arg = tuple(arg)
        elif isinstance(arg, str) and arg.isdigit():
            return int(arg)
        elif isinstance(arg, str) and arg.lower() == "true":
            arg = True
        elif isinstance(arg, str) and arg.lower() == "false":
            arg = False
        return arg

    def to_dict(self):
        data = {}
        for k, v in self:
            if hasattr(v, "_identify"):
                data[k] = v.to_dict()
            else:
                data[k] = v
        return data

    def dump_json(self, file):
        json.dump(self.to_dict(), open(file, "w"))

    def dump_yaml(self, file):
        yaml.dump(self.to_dict(), open(file, "w"), default_flow_style=False)

    @classmethod
    def load(cls, fp_name_dict: Union[str, dict]):
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, config_dict):
        config = cls()
        config.update(config_dict)
        return config

    def update(self, updates: dict):
        for k, orig_v in self:
            if k in updates:
                v = updates.pop(k)
                if isinstance(v, dict) and hasattr(orig_v, "_identify"):
                    orig_v.update(v)
                else:
                    if isinstance(orig_v, Iterable):
                        if isinstance(v, Iterable):
                            setattr(self, k, orig_v + v)
                        else:
                            setattr(self, k, orig_v.add(v))
                    else:
                        setattr(self, k, v)

    def list_subconfigs(self):
        subconfig_names = []
        for k, orig_v in self:
            if hasattr(orig_v, "_identify"):
                subconfig_names.append(k)
        return subconfig_names
