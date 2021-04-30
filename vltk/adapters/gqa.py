from collections import Counter

import vltk
from vltk import adapters
from vltk.utils.adatpers import clean_label


class GQA(adapters.VisnLangDataset):
    data_info = {
        "dev": {"coco2014": ["test"]},
        "train": {"visualgenome": ["train"]},
        "val": {"visualgenome": ["train"]},
        "test": {"coco2014": ["test"]},
        "testdev": {"coco2014": ["val"]},
    }

    def schema():
        return {}

    def forward(json_files, split, min_label_frequency=2):
        skipped = 0
        label_frequencies = Counter()
        batch_entries = []

        for t in json_files:
            for i, (k, v) in enumerate(t.items()):
                if "answer" in v:
                    answer = clean_label(v["answer"])
                    label_frequencies.update([answer])

            for i, (k, v) in enumerate(t.items()):
                if split == "test":
                    answer = None
                elif label_frequencies[v["answer"]] < min_label_frequency:
                    skipped += 1
                    continue
                else:
                    answer = clean_label(v["answer"])

                text = v["question"]
                img_id = v["imageId"].lstrip("n")
                entry = {
                    vltk.text: text,
                    vltk.imgid: img_id,
                    vltk.label: [answer],
                    vltk.score: [1.0],
                }

                batch_entries.append(entry)

        return batch_entries
