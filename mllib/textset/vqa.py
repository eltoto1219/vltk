from collections import Counter, OrderedDict, defaultdict

import datasets as ds
from mllib.abc.textset import Textset
from mllib.configs import Config
from mllib.metrics import soft_score
from tqdm import tqdm

TESTPATH = "/playpen1/home/avmendoz/data/vqa/train"


# user must only define forward in this function, and dataset features


class VQAset(Textset):
    name = "vqa"
    imageset = "coco"
    features = ds.Features(
        OrderedDict(
            {
                "img_id": ds.Value("string"),
                "text": ds.Value("string"),
                "label": ds.Sequence(length=-1, feature=ds.Value("string")),
                "score": ds.Sequence(length=-1, feature=ds.Value("float32")),
                "qid": ds.Value("string"),
            }
        )
    )

    def forward(text_data, **kwargs):
        batch_entries = []
        all_questions = []
        qid2answers = {}
        for x in tqdm(text_data):
            if "questions" in x:
                all_questions.extend(x["questions"])
            else:
                annotations = x["annotations"]
                accepted_answers = {
                    anno["multiple_choice_answer"] for anno in annotations
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
                            answer_counter.update([ans])
                    qid2answers[qid] = {
                        k: soft_score(v) for k, v in answer_counter.items()
                    }

        for entry in tqdm(all_questions):
            entry[Textset.img_key] = str(entry.pop("image_id"))
            entry[Textset.text_key] = entry.pop("question")
            entry["qid"] = str(entry.pop("question_id"))
            entry[Textset.label_key] = qid2answers[entry["qid"]]
            labels, scores = Textset._label_handler(entry[Textset.label_key])
            entry[Textset.score_key] = scores
            entry[Textset.label_key] = labels
            batch_entries.append(entry)

        return batch_entries


if __name__ == "__main__":

    config = Config().data

    # VQAset.extract(
    #     config=config,
    #     split="trainval",
    # )
    print("FORM Conf")
    loaded = VQAset.from_config(config, split="val")
    # get min frequency of answers when loading, so we know the lenngth right away
    print(loaded)

    val = loaded["val"]
    for data in val.text_iter():
        print(data)
        break
    # row = val.text_iter()

    # print(row)
    print(val.get_from_img("262148"))
    # print(val.get_freq("table"))
