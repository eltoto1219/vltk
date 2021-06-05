from collections import Counter

import datasets as ds
from tqdm import tqdm
from vltk.abc.visnlangdatasetadapter import VisnLangDatasetAdapter

# user must only define forward in this function, and dataset features


class GQAset(VisnLangDatasetAdapter):
    name = "gqa"
    data_info = {
        "dev": {"coco2014": ["test"]},
        "train": {"vg": ["train"]},
        "val": {"vg": ["train"]},
        "test": {"coco2014": ["test"]},
    }
    default_features = {
        "qid": ds.Value("string"),
        "structure": ds.Value("string"),
        "super_class": ds.Value("string"),
        "operations": ds.Sequence(length=-1, feature=ds.Value("string")),
    }
    filters = ["all"]

    def forward(text_data, split, label_preprocessor=None, **kwargs):
        skipped = 0
        min_label_frequency = kwargs.get("min_label_frequency")
        label_frequencies = Counter()
        batch_entries = []
        if label_preprocessor is None:

            def label_preprocessor(x):
                return x

        for t in text_data:
            for i, (k, v) in tqdm(enumerate(t.items())):
                if "answer" in v:
                    answer = label_preprocessor(v["answer"])
                    label_frequencies.update([answer])

            for i, (k, v) in enumerate(t.items()):
                if split == "test":
                    answer = None
                    operations = [""]
                    super_class = ""
                    structure = ""
                elif label_frequencies[v["answer"]] < min_label_frequency:
                    skipped += 1
                    continue
                else:
                    answer = label_preprocessor(v["answer"])
                    operations = [sv["operation"] for sv in v["semantic"]]
                    super_class = v["groups"]["global"]
                    structure = v["types"]["structural"]
                    if super_class is None:
                        super_class = ""

                text = v["question"]
                img_id = v["imageId"].lstrip("n")
                qid = k
                entry = {
                    "qid": qid,
                    "structure": structure,
                    "super_class": super_class,
                    "operations": operations,
                    VisnLangDatasetAdapter.text_key: text,
                    VisnLangDatasetAdapter.img_key: img_id,
                    VisnLangDatasetAdapter.label_key: [answer],
                    VisnLangDatasetAdapter.score_key: [1.0],
                }

                batch_entries.append(entry)

        print(f"SKIPPEd {skipped} entries")
        return batch_entries


if __name__ == "__main__":

    from vltk.configs import Config

    config = Config().data
    GQAset.extract(
        config=config,
        splits=["dev", "test", "val", "train"],
    )
    config.min_label_frequency = 1
    train = GQAset.from_config(config, splits="train")["train"]
    # get min frequency of answers when loading, so we know the lenngth right away

    # get path to arrow file
    # val.get_arrow_split(datadirs=config.datadirs, extractor="frcnn", split="val")
    # get path to raw img ids
    print(len(train.labels))

    # print("entry at row 1:", val.get_row(1))
    # print("entries with img id 262148:", val.get_from_img("262148"))
    # print("freq of answer table:", val.get_freq("table"))
    # print("num_labels", val.num_labels)
