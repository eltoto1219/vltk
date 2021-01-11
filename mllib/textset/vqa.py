from collections import Counter

import datasets as ds
from mllib.abc.textset import Textset
from mllib.metrics import soft_score
from tqdm import tqdm

TESTPATH = "/playpen1/home/avmendoz/data/vqa/train"


# user must only define forward in this function, and dataset features


class VQAset(Textset):
    name = "vqa"
    info = {
        "val": {"coco2014": ["val"]},
        "train": {"coco2014": ["train"]},
        "test": {"coco2014": ["test"]},
    }
    default_features = {
        "qid": ds.Value("string"),
    }

    def forward(text_data, label_processor=None, **kwargs):
        min_label_frequency = kwargs.get("min_label_frequency")
        batch_entries = []
        all_questions = []
        qid2answers = {}
        label_frequencies = Counter()
        if label_processor is None:

            def label_processor(x):
                return x

        for x in tqdm(text_data):
            if "questions" in x:
                all_questions.extend(x["questions"])
            else:
                annotations = x["annotations"]
                accepted_answers = {
                    label_processor(anno["multiple_choice_answer"])
                    for anno in annotations
                }
                for anno in tqdm(annotations):
                    qid = str(anno["question_id"])
                    answers = anno["answers"]
                    answer_counter = Counter()
                    for ans_dict in answers:
                        ans = ans_dict["answer"]
                        if ans not in accepted_answers:
                            pass
                        else:
                            ans = label_processor(ans)
                            # make  sure to clean label before updating frequncies
                            label_frequencies.update([ans])
                            answer_counter.update([ans])
                    qid2answers[qid] = {
                        k: soft_score(v) for k, v in answer_counter.items()
                    }

        skipped = 0
        for entry in tqdm(all_questions):
            entry[Textset.img_key] = str(entry.pop("image_id"))
            entry[Textset.text_key] = entry.pop("question")
            entry["qid"] = str(entry.pop("question_id"))
            entry[Textset.label_key] = qid2answers[entry["qid"]]
            labels = {
                l: s
                for l, s in entry[Textset.label_key].items()
                if label_frequencies[l] > min_label_frequency
            }
            if not labels:
                skipped += 1
                continue

            labels, scores = Textset._label_handler(labels)
            entry[Textset.score_key] = scores
            entry[Textset.label_key] = labels
            batch_entries.append(entry)

        print(f"SKIPPEd {skipped} entries")
        return batch_entries

    # from mllib.configs import Config
    # config = Config().data
    # save_to
    # path_or_dir

    # VQAset.extract(
    #     config=config,
    #     split="trainval",
    # )
    # loaded = VQAset.from_config(config, split="val")
    # # get min frequency of answers when loading, so we know the lenngth right away
    # print(loaded)
    # val = loaded["val"]

    # print("entry at row 1:", val.get_row(1))
    # print("entries with img id 262148:", val.get_from_img("262148"))
    # print("freq of answer table:", val.get_freq("table"))
    # print("num_labels", val.num_labels)
