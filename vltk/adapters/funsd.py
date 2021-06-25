import json
import os

import cv2
import matplotlib.pyplot as plt
import vltk
from tqdm import tqdm
from vltk import Features, adapters


class FUNSD(adapters.VisnDataset):
    def schema():
        return {
            vltk.tokenbox: Features.Box,
            vltk.text: Features.StringList,
            vltk.label: Features.StringList,
            # "linking": Features.NestedIntList,
        }

    def forward(json_files, splits, datadir=None):
        imgids = set()
        annos = []
        for filename, data in tqdm(json_files.items()):
            text = []
            words = []
            labels = []
            boxes = []
            linkings = []
            imgid = filename.split(".")[0]
            assert imgid not in imgids
            imgids.update([imgid])

            for item in data["form"]:
                label = item["label"]
                linking = item["linking"]
                if not linking:
                    linking = [[0, 0]]
                words = item["words"]
                labels += [label] * len(words)
                linkings += linking * len(words)
                for word in words:
                    text.append(word["text"])
                    boxes.append(word["box"])

            entry = {
                vltk.text: text,
                vltk.tokenbox: boxes,
                vltk.label: labels,
                vltk.imgid: str(imgid),
                # "linking": linkings,
            }
            annos.append(entry)

        return annos
