import json
import os

import cv2
import matplotlib.pyplot as plt
import vltk.vars as vltk
from tqdm import tqdm
from vltk import adapters
from vltk.features import Features


class FUNSD(adapters.VisnDataset):

    urls = "https://guillaumejaume.github.io/FUNSD/dataset.zip"

    @staticmethod
    def schema():
        return {
            vltk.tokenbox: Features.Box(),
            vltk.text: Features.StringList(),
            vltk.label: Features.StringList(),
            # "linking": Features.NestedIntList,
        }

    @staticmethod
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
                if label not in ("question", "answer", "other"):
                    label = "other"
                linking = item["linking"]
                if not linking:
                    linking = [[0, 0]]
                words = item["words"]
                labels += [label] * len(words)
                linkings += linking * len(words)

                for word in words:
                    text.append(word["text"])
                    x1, y1, x2, y2 = word["box"]
                    boxes.append([x1, y1, x2 - x1, y2 - y1])

            assert len(labels) == len(text)

            entry = {
                vltk.text: text,
                vltk.tokenbox: boxes,
                vltk.label: labels,
                vltk.imgid: str(imgid),
            }
            annos.append(entry)

        return annos
