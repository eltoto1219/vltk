import json
import os
from collections import OrderedDict

import cv2
import jsonlines
import numpy as np
import torch
import torch.nn.functional as F
from mllib.utils import get_subfiles_from_path

import datasets

__all__ = ["NAME2DATASET"]

ANS_CONVERT = {
    "a man": "man",
    "the man": "man",
    "a woman": "woman",
    "the woman": "woman",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "grey": "gray",
}


# let us try plotting the exapnded/condensed image before we try anything weird
def img_to_tensor(
    fp, min_size=832, max_size=832, pad_value=0, mean=None, sdev=None, use_gpu=False
):
    assert isinstance(fp, str)
    assert os.path.isfile(fp)
    img = cv2.imread(fp)
    if img is None:
        return None, (None, None), (None, None)
    img = img[:, :, ::1]
    img = torch.as_tensor(img).float()
    if use_gpu:
        img = img.cuda()
    h, w = img.shape[:2]
    scale = min_size * 1.0 / min(h, w)
    if h < w:
        newh, neww = min_size, scale * w
    else:
        newh, neww = scale * h, min_size
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)

    img = img.permute(2, 0, 1).unsqueeze(0)  # 3, 0, 1)  # hw(c) -> nchw
    img = F.interpolate(
        img, (newh, neww), mode="bilinear", align_corners=False
    ).squeeze(0)
    img = torch.clamp(img, max=255)
    if mean is not None and sdev is not None:
        img = (img - mean) / sdev
    if pad_value is not None:
        size = img.shape[-2:]
        img = F.pad(
            img,
            [0, max_size - size[1], 0, max_size - size[0]],
            value=pad_value,
        )
    if use_gpu:
        img = img.cpu()
    return img, (h, w), (newh, neww)


"""
    def matched_sentence_modeling(self, entry):
        is_matched = 1
        if random.random() < self.dataset_config.sentence_match_rate:
            is_matched = 0
            other_datum = self.text_data[random.randint(0, len(self.text_data) - 1)]
            while other_datum["img_id"] == entry["img_id"]:
                other_datum = self.text_data[random.randint(0, len(self.text_data) - 1)]
            sent = other_datum["text"]
            entry["text"] = sent
        if not is_matched:
            entry["label"] = torch.tensor(self.ignore_id)

        is_matched = torch.tensor(is_matched)
        entry["is_matched"] = is_matched
        return entry

    def masked_language_modeling(self, input_ids):
        ignore_id = self.ignore_id
        mask_id = self.mask_id
        masked_sequence = input_ids
        masked_inds = torch.zeros(input_ids.shape)
        # do random_sampling instead of iteration: faster
        for i in range(len(masked_sequence)):
            ind = int(masked_sequence[i])
            random_id = self.random_ind()
            while random_id in self.special_ids:
                random_id = self.random_ind()
            prob = random.random()
            ratio = self.dataset_config.word_mask_rate
            if prob < ratio and ind not in self.special_ids:
                prob /= ratio
                if prob < 0.8:
                    masked_sequence[i] = mask_id
                elif prob < 0.9:
                    masked_sequence[i] = random_id
                assert ind not in self.special_ids
                masked_inds[i] = ind
            else:
                masked_inds[i] = ignore_id

        return masked_inds, masked_sequence

    def get_random_ind(self):
        token = random.choice(list(self.tokenizer.vocab.items()))
        return int(token[-1])

"""


def process_answer_default(ans):
    if len(ans) == 0:
        return ""
    ans = ans.lower()
    if ans[-1] == ".":
        ans = ans[:-1].strip()
    if ans.startswith("a "):
        ans = ans[2:].strip()
    if ans.startswith("an "):
        ans = ans[3:].strip()
    if ans.startswith("the "):
        ans = ans[4:].strip()
    if ans in ANS_CONVERT:
        ans = ANS_CONVERT[ans]
    return ans


def load_temp_gqa(config, split):
    use_raw = config.data.use_raw_imgs
    arrow_dict = None
    path_dict = None
    data_dir = config.pathes.data_dir

    # arrow file
    fields = config.data.arrow_fields
    if fields is None or fields:
        if split not in ("inference", "dev", "testdev", "eval", "evaluation"):
            vg = datasets.Dataset.from_file(
                os.path.join(data_dir, config.pathes.vg_train_arrow)
            )

        else:
            vg = datasets.Dataset.from_file(
                os.path.join(data_dir, config.pathes.vg_test_arrow)
            )
        if fields is not None:
            fields = list(fields)
        vg.set_format(type="numpy", columns=fields)
        arrow_dict = {"gqa": vg}
    else:
        arrow_dict = None

    # raw tensors
    if use_raw:
        if split in ("testdev", "test", "eval", "dev"):
            path_dict = {
                "gqa": os.path.join(data_dir, config.pathes.vg_test),
            }
        else:
            path_dict = {
                "gqa": os.path.join(data_dir, config.pathes.vg_train),
            }

    # labels
    labels = json.load(open(os.path.join(data_dir, config.pathes.gqa_labels)))

    # text data
    if split in ("pretrain", "train", "finetune"):
        text_files = get_subfiles_from_path(config.pathes.gqa_train, relative=data_dir)
    elif split in (
        "validation",
        "val",
        "valid",
    ):
        text_files = get_subfiles_from_path(config.pathes.gqa_val, relative=data_dir)
    elif split in ("inference", "dev", "testdev", "eval", "evaluation"):
        text_files = get_subfiles_from_path(
            config.pathes.gqa_testdev, relative=data_dir
        )
    else:
        text_files = get_subfiles_from_path(config.pathes.gqa_test, relative=data_dir)

    print(f"spits to be loaded: {text_files}")
    ignored_labels = set()
    num_ignored = 0
    text_data = []
    for text in text_files:
        try:
            data_split = json.load(open(text))
        except json.decoder.JSONDecodeError:
            data_split = jsonlines.open(text)
        for data in data_split:
            img_id = data["img_id"].split("_")[-1]
            entry = {"img_id": img_id, "text": data["sent"], "dset": "gqa"}
            label = data["label"]
            if isinstance(label, dict):
                if len(label) == 0:
                    continue
            elif isinstance(label, str):
                label = {label: 1.0}
                label = next(iter(data["label"].keys()))
            assert len(label) == 1
            for l, s in label.items():
                l = process_answer_default(l)
                if l not in labels:
                    num_ignored += 1
                    ignored_labels.add(l)
                    continue
                else:
                    entry["label"] = torch.tensor(labels[l])
                    text_data.append(entry)
            if config.dryrun:
                break
        if config.dryrun:
            break
    print(f"num ignored: {num_ignored} ignored: {ignored_labels}")
    return text_data, path_dict, arrow_dict, labels


def load_temp_lxmert(config, split):
    data_dir = config.pathes.data_dir

    coco_valid = datasets.Dataset.from_file(
        os.path.join(data_dir, config.pathes.coco_valid_arrow)
    )
    coco_train = datasets.Dataset.from_file(
        os.path.join(data_dir, config.pathes.coco_train_arrow)
    )
    coco = datasets.concatenate_datasets([coco_valid, coco_train])
    coco.set_format(type="numpy", output_all_columns=True)
    vg = datasets.Dataset.from_file(os.path.join(data_dir, config.pathes.vg_arrow))
    vg.set_format(type="numpy", output_all_columns=True)
    arrow_dict = {"coco": coco, "vg": vg}

    path_dict = {
        "coco": os.path.join(data_dir, config.pathes.coco_imgs),
        "vg": os.path.join(data_dir, config.pathes.vg_imgs),
    }

    if split in ("pretrain", "train", "finetune"):
        text_files = get_subfiles_from_path(
            config.pathes.temp_lxmert_train, relative=data_dir
        )
    elif split in ("eval", "evaluation", "validation", "val"):
        text_files = get_subfiles_from_path(
            config.pathes.temp_lxmert_eval, relative=data_dir
        )
    else:
        text_files = get_subfiles_from_path(
            config.pathes.temp_lxmert_test, relative=data_dir
        )

    labels = set(
        [
            process_answer_default(ans["ans"])
            for ans in json.load(
                open(os.path.join(data_dir, config.pathes.temp_lxmert_answers))
            )
        ]
    )

    name2dset = {
        "mscoco": "coco",
        "vg": "vg",
        "visual7w": "vg",
        "gqa": "vg",
        "vqa": "coco",
    }
    print("loading text data")
    num_labels = 0
    label2lid = {}
    ignore_idx = -100
    text_data = []
    for text in text_files:
        data_split = json.load(open(text))
        for data in data_split:
            img_id = data["img_id"].split("_")[-1]
            sentf = data["sentf"]
            for sents_cat, sents in sentf.items():
                dset = name2dset[sents_cat]
                if sents_cat in data["labelf"]:
                    labels = data["labelf"][sents_cat]
                else:
                    labels = None
                for sent_idx, sent in enumerate(sents):
                    entry = {"img_id": img_id, "text": sent, "dset": dset}
                    if labels is not None:
                        label = labels[sent_idx]
                        in_label_set = False
                        if not len(label) == 0:
                            label = OrderedDict(
                                {
                                    k: v
                                    for k, v in sorted(
                                        label.items(),
                                        key=lambda item: item[1],
                                        reverse=True,
                                    )
                                }
                            )
                            for k in label.keys():
                                k = process_answer_default(k)
                                if k in labels:
                                    in_label_set = True
                                    label = k
                                    break
                            if in_label_set:
                                if label not in labels:
                                    lid = num_labels
                                    label2lid[label] = lid
                                    num_labels += 1
                                else:
                                    lid = label2lid[label]
                            else:
                                lid = ignore_idx
                    else:
                        lid = ignore_idx
                    entry["label"] = torch.tensor(lid)
                    text_data.append(entry)

    return text_data, path_dict, arrow_dict, labels


NAME2DATASET = {"temp_lxmert": load_temp_lxmert, "gqa": load_temp_gqa}


if __name__ == "__main__":
    fp = "/playpen1/home/avmendoz/data/coco/test2017"
    ex = next(iter(map(lambda x: os.path.join(fp, x), os.listdir(fp))))
    ex = img_to_tensor(ex)[0]
    print(cv2.imwrite("./test.jpg", ex.permute(1, 2, 0).numpy().astype(np.uint8)))
