from collections import Counter

import vltk.vars as vltk
from vltk import adapters
from vltk.adapters import Adapters
from vltk.features import Features
from vltk.utils.adapters import clean_label


class VGQA(adapters.VisnDataset):
    data_info = {
        "train": {"visualgenome": ["train"]},
    }

    def schema():
        # img id and score are assumed to be default features
        return {
            vltk.qid: Features.String(),
            vltk.label: Features.StringList(),
        }

    def adjust_imgid(imgid, vdset_name, vdset_split):
        # length of COCO ids are are always length 12
        # imgid = f'{"COCO"}_{vdset_split[0].lower()}{2014}_{"".join(["0"] * (12 - len(imgid)))}{imgid}'
        return imgid

    def forward(json_files, split, min_label_frequency=9):
        batch_entries = []
        answer_counts = Counter()
        skipped = 0
        for filename, data in json_files.items():
            for y in data:
                for x in y["qas"]:
                    answer_counts.update([clean_label(x["answer"])]),

        for filename, data in json_files.items():
            for y in data:
                for x in y["qas"]:
                    if answer_counts[clean_label(x["answer"])] >= min_label_frequency:
                        entry = {
                            vltk.qid: str(x["qa_id"]),
                            vltk.imgid: str(x["image_id"]),
                            vltk.text: x["question"],
                            vltk.label: [clean_label(x["answer"])],
                        }
                        batch_entries.append(entry)
                    else:
                        skipped += 1
        print(
            f"num skipped: {skipped / len(batch_entries)}, unfilterd num answers: {len(answer_counts)}"
        )
        return batch_entries

    # first entry in the dataset
