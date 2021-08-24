import json
import os
from collections import defaultdict

import vltk.vars as vltk
from tqdm import tqdm
from vltk import adapters
from vltk.features import Features
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
    # filters = ["train"]

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

    def schema():
        return {
            "answer": Features.String(),
            vltk.qid: Features.String(),
            vltk.span: Features.IntList(),
        }

    def forward(json_files, split, datadir=None):
        skipped = 0
        batch_entries = []
        data = defaultdict(list)
        for filename, item in json_files.items():
            data = item["data"]
            for d in tqdm(data):
                question = d["question"].lower().replace('"', "")
                image = d["image"]
                docid = d["docId"]
                # split = d["split"]
                imgid = image.split(".")[0].split("/")[-1]
                # open annotation:
                answers = list(map(lambda x: x.lower(), d["answers"]))
                # if this is not the correct path to the annotation,  change the path here
                anno = json.load(
                    open(
                        os.path.join(
                            datadir,
                            "docvqavisn",
                            "annotations",
                            f"{imgid}.json",
                        ),
                        "r",
                    )
                )["recognitionResults"][0]

                words = ()
                for lines in anno["lines"]:
                    for word in lines["words"]:
                        words += (word["text"].lower(),)

                if not words:
                    skipped += 1
                    continue

                inds, max_jaccard, keep_answer = get_span_via_jaccard(
                    words,
                    answers,
                )
                if inds[0] is None:
                    skipped += 1
                    continue
                if inds[0] == inds[1]:
                    answer_in_doc = words[inds[0]]
                else:
                    answer_in_doc = " ".join(words[inds[0] : inds[1]])
                if max_jaccard < 0.56:
                    skipped += 1
                    continue
                # if 0.5 <= max_jaccard <= 0.6:
                #     print(f"j: {max_jaccard}, a: {keep_answer}, ~: {answer_in_doc}")
                #     print(answers)

                entry = {
                    vltk.text: question,
                    vltk.imgid: imgid,
                    "answer": answer_in_doc,
                    vltk.span: list(inds),
                    vltk.qid: str(docid),
                }
                batch_entries.append(entry)
        print(f"skipped {skipped} questions: could not find answer.")
        return batch_entries


class DocVQAVisn(adapters.VisnDataset):
    def schema():
        return {
            vltk.box: Features.Box(),
            vltk.tokenbox: Features.Box(),
            vltk.text: Features.StringList(),
        }

    def forward(json_files, splits, datadir=None):
        imgids = set()
        annos = []
        for filename, data in tqdm(json_files.items()):
            entry = {}
            imgid = filename.split(".")[0].split("/")[-1]
            assert imgid not in imgids
            imgids.add(imgid)
            if "status" in data:
                status = 1 if data["status"] == "Succeeded" else 0
                if status == 0:
                    continue
            else:
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
                    box = DocVQA.format_box(lines["boundingBox"])
                    assert len(box) == 4
                    texts.append(text)
                    tokenboxes.append(box)
            if not texts:
                continue
            if isinstance(texts[0], list):
                raise Exception(texts)
            entry = {
                vltk.imgid: imgid,
                vltk.box: boxes,
                vltk.text: texts,
                vltk.tokenbox: tokenboxes,
            }
            annos.append(entry)

        return annos
