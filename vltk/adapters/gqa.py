from collections import Counter

from vltk import Features, adapters
from vltk.utils.adapters import clean_label
from vltk.vars import Vars as vltk


class GQA(adapters.VisnLangDataset):
    data_info = {
        "dev": {"coco2014": ["test"]},
        "train": {"visualgenome": ["train"]},
        "val": {"visualgenome": ["train"]},
        "test": {"coco2014": ["test"]},
        "testdev": {"coco2014": ["val"]},
    }

    filters = ["unbalanced", "train"]

    @staticmethod
    def schema():
        return {vltk.label: Features.StringList(), "layout": Features.StringList()}

    @staticmethod
    def forward(json_files, split, min_label_frequency=2):
        skipped = 0
        label_frequencies = Counter()
        batch_entries = []

        for filename, data in json_files.items():
            for i, (k, v) in enumerate(data.items()):
                if "answer" in v:
                    answer = clean_label(v["answer"])
                    label_frequencies.update([answer])

            for i, (k, v) in enumerate(data.items()):
                if split == "test":
                    answer = None
                    layout = None
                elif label_frequencies[v["answer"]] < min_label_frequency:
                    skipped += 1
                    continue
                else:
                    answer = clean_label(v["answer"])
                    layout = [layout["operation"] for layout in v["semantic"]]

                text = v["question"]
                img_id = v["imageId"].lstrip("n")

                entry = {
                    vltk.text: text,
                    vltk.imgid: img_id,
                    vltk.label: [answer],
                    "layout": layout,
                }

                batch_entries.append(entry)

        return batch_entries
