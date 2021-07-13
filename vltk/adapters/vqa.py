from collections import Counter

import vltk.vars as vltk
from vltk import adapters
from vltk.features import Features
from vltk.utils.adapters import clean_label, soft_score


# Vision-Language Datasets
class VQA(adapters.VisnLangDataset):
    data_info = {
        "val": {"coco2014": ["val"]},
        "train": {"coco2014": ["train"]},
        "test": {"coco2014": ["test"]},
    }

    @staticmethod
    def schema():
        # img id and score are assumed to be default features
        return {
            vltk.qid: Features.String(),
            vltk.label: Features.StringList(),
            vltk.score: Features.FloatList(),
        }

    @staticmethod
    def adjust_imgid(imgid, vdset_name, vdset_split):
        # length of COCO ids are are always length 12
        imgid = f'{"COCO"}_{vdset_split[0].lower()}{2014}_{"".join(["0"] * (12 - len(imgid)))}{imgid}'
        return imgid

    @staticmethod
    def forward(json_files, split, min_label_frequency=9):
        batch_entries = []
        all_questions = []
        qid2answers = {}
        label_frequencies = Counter()
        for filename, x in json_files.items():
            if "questions" in x:
                all_questions.extend(x["questions"])
            else:
                annotations = x["annotations"]
                accepted_answers = {
                    clean_label(anno["multiple_choice_answer"]) for anno in annotations
                }
                for anno in annotations:
                    qid = str(anno["question_id"])
                    answers = anno["answers"]
                    label_frequencies.update(
                        [clean_label(anno["multiple_choice_answer"])]
                    )
                    answer_counter = Counter()
                    for ans_dict in answers:
                        ans = ans_dict["answer"]
                        if ans not in accepted_answers:
                            pass
                        else:
                            ans = clean_label(ans)
                            answer_counter.update([ans])
                    qid2answers[qid] = {
                        k: soft_score(v) for k, v in answer_counter.items()
                    }

        skipped = 0
        for entry in all_questions:
            try:
                entry[vltk.imgid] = str(entry.pop("image_id"))
            except Exception:
                raise Exception(entry.keys())
            entry[vltk.text] = entry.pop("question")
            # entry.pop("question_id")
            entry["qid"] = str(entry.pop("question_id"))
            try:
                entry[vltk.label] = qid2answers[entry["qid"]]
                labels = {
                    l: s
                    for l, s in entry[vltk.label].items()
                    if label_frequencies[l] > min_label_frequency
                }
                if not labels:
                    skipped += 1
                    continue

                labels, scores = adapters.VisnLangDataset._label_handler(labels)
                entry[vltk.score] = scores
                entry[vltk.label] = labels
            except KeyError:
                pass

            batch_entries.append(entry)
        return batch_entries
