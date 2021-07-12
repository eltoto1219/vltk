import json
import os

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from vltk import Features, adapters
from vltk.vars import Vars as vltk


class FUNSD(adapters.VisnDataset):
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

            # try:
            #     imgfile = os.path.join(datadir, "funsd", f"train/{imgid}.png")
            # except Exception:
            #     imgfile = os.path.join(datadir, "funsd", f"test/{imgid}.png")
            # img = cv2.imread(imgfile)

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

                # color = (255, 255, 255)
                # if label == "question":
                #     color = (255, 0, 0)
                # elif label == "answer":
                #     color = (0, 255, 0)

                for word in words:
                    text.append(word["text"])
                    x1, y1, x2, y2 = word["box"]
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
                    # img = cv2.rectangle(
                    #     img,
                    #     (word["box"][0], word["box"][1]),
                    #     (word["box"][2], word["box"][3]),
                    #     color,
                    # )
            # if img is not None:
            #     cv2.imwrite(f"/home/eltoto/exs_funsd/{imgid}.png", img)
            assert len(labels) == len(text)

            entry = {
                vltk.text: text,
                vltk.tokenbox: boxes,
                vltk.label: labels,
                vltk.imgid: str(imgid),
                # "linking": linkings,
            }
            annos.append(entry)

        return annos
