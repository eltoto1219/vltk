import json
import os

from vltk import LABELPROCPATH
from vltk.inspection import import_funcs_from_file

PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "libdata"
)
ANS_CONVERT = json.load(open(os.path.join(PATH, "convert_answers.json")))
CONTRACTION_CONVERT = json.load(open(os.path.join(PATH, "convert_answers.json")))


class Label:
    def __init__(self):
        if "LABELPROCDICT" not in globals():
            global LABELPROCDICT
            LABELPROCDICT = import_funcs_from_file(LABELPROCPATH, pkg="vltk.processing")

    def avail(self):
        return list(LABELPROCDICT.keys())

    def get(self, name):
        return LABELPROCDICT[name]

    def add(self, name, lab):
        LABELPROCDICT[name] = lab


def clean_imgid_default(imgid):
    return imgid.split("_")[-1].lstrip("0").strip("n")
    # started = False
    # new_id = ""
    # for i in imgid:
    #     if not i.isalnum():
    #         started = True
    #         new_id += i
    #     else:
    #         if started:
    #             return new_id
    #         else:
    #             pass


def label_default(ans):
    if len(ans) == 0:
        return ""
    ans = ans.lower()
    ans = ans.replace(",", "")
    if ans[-1] == ".":
        ans = ans[:-1].strip()
    if ans.startswith("a "):
        ans = ans[2:].strip()
    if ans.startswith("an "):
        ans = ans[3:].strip()
    if ans.startswith("the "):
        ans = ans[4:].strip()
    ans = " ".join(
        [
            CONTRACTION_CONVERT[a] if a in CONTRACTION_CONVERT else a
            for a in ans.split(" ")
        ]
    )
    if ans in ANS_CONVERT:
        ans = ANS_CONVERT[ans]
    return ans
