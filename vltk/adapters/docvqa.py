import json
import os

import vltk
from tqdm import tqdm
from vltk import Features, adapters
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

    def schema():
        # img id, label, and score are assumed to be default features
        return {vltk.span: Features.Span}

    def forward(json_files, split, datadir=None):
        skipped = 0
        batch_entries = []
        for filename, item in json_files.items():
            data = item["data"]
            for d in tqdm(data):
                question = d["question"]
                image = d["image"]
                # docid = d["docId"]
                imgid = image.split(".")[0].split("/")[-1]
                # open annotation:

                answers = list(map(lambda x: x.lower(), d["answers"]))
                try:
                    anno = json.load(
                        open(
                            os.path.join(
                                datadir, "docvqavisn/annotations", f"{imgid}.json"
                            ),
                            "r",
                        )
                    )["recognitionResults"][0]
                except FileNotFoundError:
                    skipped += 1
                    continue

                words = ()
                for lines in anno["lines"]:
                    for word in lines["words"]:
                        words += (word["text"].lower(),)
                if not words:
                    skipped += 1
                    continue

                span, max_jaccard = get_span_via_jaccard(words, answers, skipped)
                if span is None:
                    continue

                entry = {vltk.text: question, vltk.imgid: imgid, vltk.span: span}
                batch_entries.append(entry)
        print(f"skipped {skipped} questions: could not find answer.")
        return batch_entries


class DocVQAVisn(adapters.VisnDataset):
    def schema():
        return {
            vltk.box: Features.Box,
            vltk.tokenbox: Features.Box,
            vltk.text: Features.StringList,
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
