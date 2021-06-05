from collections import Counter

import datasets as ds
from tqdm import tqdm
from vltk.abc.visnlangdatasetadapter import VisnLangDatasetAdapter
from vltk.metrics import soft_score

TESTPATH = "/playpen1/home/avmendoz/data/vqa/train"


# user must only define forward in this function, and dataset features


class VQAset(VisnLangDatasetAdapter):
    name = "vqa"
    data_info = {
        "val": {"coco2014": ["val"]},
        "train": {"coco2014": ["train"]},
        "test": {"coco2014": ["test"]},
    }
    default_features = {
        "qid": ds.Value("string"),
    }

    def forward(text_data, split, label_preprocessor=None, **kwargs):
        min_label_frequency = kwargs.get("min_label_frequency")
        batch_entries = []
        all_questions = []
        qid2answers = {}
        label_frequencies = Counter()
        if label_preprocessor is None:

            def label_preprocessor(x):
                return x

        for x in tqdm(text_data):
            if "questions" in x:
                all_questions.extend(x["questions"])
            else:
                annotations = x["annotations"]
                accepted_answers = {
                    label_preprocessor(anno["multiple_choice_answer"])
                    for anno in annotations
                }
                for anno in tqdm(annotations):
                    qid = str(anno["question_id"])
                    answers = anno["answers"]
                    label_frequencies.update(
                        [label_preprocessor(anno["multiple_choice_answer"])]
                    )
                    answer_counter = Counter()
                    for ans_dict in answers:
                        ans = ans_dict["answer"]
                        if ans not in accepted_answers:
                            pass
                        else:
                            ans = label_preprocessor(ans)
                            # make  sure to clean label before updating frequncies
                            # label_frequencies.update([ans])
                            answer_counter.update([ans])
                    qid2answers[qid] = {
                        k: soft_score(v) for k, v in answer_counter.items()
                    }

        skipped = 0
        for entry in tqdm(all_questions):
            entry[VisnLangDatasetAdapter.img_key] = str(entry.pop("image_id"))
            entry[VisnLangDatasetAdapter.text_key] = entry.pop("question")
            entry["qid"] = str(entry.pop("question_id"))
            try:
                entry[VisnLangDatasetAdapter.label_key] = qid2answers[entry["qid"]]
                labels = {
                    l: s
                    for l, s in entry[VisnLangDatasetAdapter.label_key].items()
                    if label_frequencies[l] > min_label_frequency
                }
                if not labels:
                    skipped += 1
                    continue

                labels, scores = VisnLangDatasetAdapter._label_handler(labels)
                entry[VisnLangDatasetAdapter.score_key] = scores
                entry[VisnLangDatasetAdapter.label_key] = labels
            except KeyError:
                pass

            batch_entries.append(entry)

        print(f"SKIPPEd {skipped} entries")
        return batch_entries


if __name__ == "__main__":

    from vltk.configs import Config

    config = Config().data
    config.min_label_frequency = 9
    VQAset.extract(
        config=config,
        splits=["train", "val", "test"],
    )
    val = VQAset.from_config(config, splits="val")["val"]
    # get min frequency of answers when loading, so we know the lenngth right away

    # get path to arrow file
    # val.get_arrow_split(datadirs=config.datadirs, extractor="frcnn", split="val")
    # get path to raw img ids
    print(len(val.labels))

    # print("entry at row 1:", val.get_row(1))
    # print("entries with img id 262148:", val.get_from_img("262148"))
    # print("freq of answer table:", val.get_freq("table"))
    # print("num_labels", val.num_labels)
