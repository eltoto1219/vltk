import json
import os
from collections import defaultdict

import numpy as np
import vltk.vars as vltk
from tqdm import tqdm
from vltk.features import Features
from vltk import adapters
from vltk.utils.adapters import get_span_via_jaccard

"""
OCR Results

'status': str,
'recognitionResults': list (each item is a page)
    'page': int, -> keep
    'clockwiseOrientation': float, -> discard
    'width': int, -> discard
    'height': int,-> discard
    'unit': string, -> discard
        'lines': list (each item is a component)
        'boundingBox': list of ints,
        'text': string,
        'words':
            'boundingBox': list of ints,
                        'text': str,
                                'confidence': str  (optional)
"""


class DocVQA(adapters.VisnLangDataset):
    data_info = {
        "val": {"docvqavisn": ["val"]},
        "train": {"docvqavisn": ["train"]},
    }

    @staticmethod
    def format_box(box):
        x1, y1, x2, y2, x3, y3, x4, y4 = box
        new_x1 = min([x1, x2, x3, x4])
        new_x2 = max([x1, x2, x3, x4])
        new_y1 = min([y1, y2, y3, y4])
        new_y2 = max([y1, y2, y3, y4])
        width = abs(new_x2 - new_x1)
        height = abs(new_y2 - new_y1)
        return [x1, y1, width, height]

    @staticmethod
    def schema():
        # img id, label, and score are assumed to be default features
        return {vltk.label: Features.StringList()}  # vltk.span: Features.Span,

    @staticmethod
    def forward(json_files, split, datadir=None):
        total = 0
        skipped = 0
        batch_entries = []
        dataa = defaultdict(list)
        for filename, item in json_files.items():
            data = item["data"]
            for d in tqdm(data):
                split = d["data_split"]
                question = d["question"].lower().replace('"', "")
                cont = True
                for x in (
                    "location",
                    "address",
                    "site",
                    "city",
                    "state",
                    "zip code",
                    "street",
                ):
                    if (
                        x in question
                        and "email" not in question
                        and "value" not in question
                        and "how much" not in question
                        and "percentage" not in question
                        and "to who" not in question
                        and "does this" not in question
                        and "what type" not in question
                        and "whom" not in question
                        and "when" not in question
                        and "last, first" not in question
                        and "date" not in question
                        and "name of a" not in question
                        and "name of c" not in question
                        and "name of m" not in question
                        and "addressed" not in question
                        and "is the drug" not in question
                        and "modifications" not in question
                        and "who" not in question
                        and "website" not in question
                        and len(question.split()) < 12
                    ):
                        cont = False
                        print(question)
                        total += 1
                        break
                if cont:
                    continue
                image = d["image"]
                # docid = d["docId"]
                imgid = image.split(".")[0].split("/")[-1]
                # open annotation:
                fileid = imgid + ".png"

                answers = list(map(lambda x: x.lower(), d["answers"]))
                anno = json.load(
                    open(
                        os.path.join(
                            datadir,
                            "docvqavisn",
                            split,
                            "ocr_results",
                            f"{imgid}.json",
                        ),
                        "r",
                    )
                )["recognitionResults"][0]

                words = ()
                answers = [answers[0]]
                for lines in anno["lines"]:
                    for word in lines["words"]:
                        words += (word["text"].lower(),)

                # if not words:
                #     skipped += 1
                #     continue

                span, max_jaccard = get_span_via_jaccard(words, answers, skipped)
                if span is not None:
                    dataa[fileid].append(span)
                # if span is None:
                #     continue

                entry = {
                    vltk.text: question,
                    vltk.imgid: imgid,
                    vltk.label: answers
                    # vltk.span: span
                }
                batch_entries.append(entry)
        print(f"skipped {skipped} questions: could not find answer.")
        json.dump(dataa, open(f"docvqa_{split}.json", "w"))
        raise Exception(total)
        return batch_entries


class DocVQAVisn(adapters.VisnDataset):
    @staticmethod
    def schema():
        return {
            vltk.box: Features.Box(),
            vltk.tokenbox: Features.Box(),
            vltk.text: Features.StringList(),
        }

    @staticmethod
    def format_box(box):
        x1, y1, x2, y2, x3, y3, x4, y4 = box
        new_x1 = min([x1, x2, x3, x4])
        new_x2 = max([x1, x2, x3, x4])
        new_y1 = min([y1, y2, y3, y4])
        new_y2 = max([y1, y2, y3, y4])
        width = abs(new_x2 - new_x1)
        height = abs(new_y2 - new_y1)
        return [x1, y1, width, height]

    @staticmethod
    def forward(json_files, splits, datadir=None):
        imgids = set()
        annos = []
        for filename, data in tqdm(json_files.items()):
            entry = {}
            imgid = filename.split(".")[0].split("/")[-1]
            assert imgid not in imgids
            imgids.add(imgid)
            status = 1 if data["status"] == "Succeeded" else 0
            if status == 0:
                continue
            data = data["recognitionResults"]
            if len(data) != 1:
                raise Exception(len(data))
            data = data[0]
            boxes = []
            tokenboxes = []
            texts = []
            for lines in data["lines"]:
                box = DocVQAVisn.format_box(lines["boundingBox"])
                boxes.append(box)
                for word in lines["words"]:
                    text = word["text"]
                    box = word["boundingBox"]
                    box = DocVQAVisn.format_box(lines["boundingBox"])
                    texts.append(text)
                    tokenboxes.append(box)
            entry = {
                vltk.imgid: imgid,
                vltk.box: boxes,
                vltk.text: texts,
                vltk.tokenbox: tokenboxes,
            }
            annos.append(entry)

        return annos
