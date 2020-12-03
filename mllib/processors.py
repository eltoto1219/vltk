import json
import os

import cv2
import datasets
import numpy as np
import torch
import torch.nn.functional as F

from .utils import get_file_path

__all__ = [
    "process_answer_default",
    "img_to_tensor",
    "load_temp_gqa",
    "load_temp_lxmert",
]

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


# let us try plotting the condensed image before we try anything weird
def img_to_tensor(fp, input_format="bgr", min_size=640, max_size=640, pad_value=0):
    assert isinstance(fp, str)
    assert os.path.isfile(fp)
    img = cv2.imread(fp)
    if img is None:
        return
    img = img[:, :, ::1]
    img = torch.as_tensor(img).float()
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
    print(img.shape)
    # size = img.shape[-2:]
    # img = F.pad(
    #     img,
    #     [0, max_size - size[1], 0, max_size - size[0]],
    #     value=pad_value,
    # )
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


def load_temp_gqa(
    pathes_config,
    global_config,
    dataset_config,
    split,
):
    data_dir = global_config.data_dir
    if split not in ("inference", "dev", "testdev", "eval", "evaluation"):
        vg = datasets.Dataset.from_file(
            os.path.join(data_dir, pathes_config.vg_train_arrow)
        )
    else:
        vg = datasets.Dataset.from_file(
            os.path.join(data_dir, pathes_config.vg_test_arrow)
        )
    arrow_dict = {"gqa": vg}

    img_dirs = {
        "gqa": [
            os.path.join(data_dir, pathes_config.vg_imgs),
            os.path.join(data_dir, pathes_config.coco_test_imgs),
        ]
    }

    if split in ("pretrain", "train", "finetune"):
        files = get_file_path(data_dir, pathes_config.gqa_train)
    elif split in (
        "validation",
        "val",
        "valid",
    ):
        files = get_file_path(data_dir, pathes_config.gqa_val)
    elif split in ("inference", "dev", "testdev", "eval", "evaluation"):
        files = get_file_path(data_dir, pathes_config.gqa_testdev)
    else:
        files = get_file_path(data_dir, pathes_config.gqa_test)

    print(f"spits to be loaded: {files}")

    labels = os.path.join(data_dir, pathes_config.gqa_labels)
    label_set = json.load(open(labels))

    return files, img_dirs, arrow_dict, label_set


def load_temp_lxmert(
    pathes_config,
    global_config,
    dataset_config,
    split,
):
    data_dir = global_config.data_dir

    coco_valid = datasets.Dataset.from_file(
        os.path.join(data_dir, pathes_config.coco_valid_arrow)
    )
    coco_train = datasets.Dataset.from_file(
        os.path.join(data_dir, pathes_config.coco_train_arrow)
    )
    coco = datasets.concatenate_datasets([coco_valid, coco_train])
    vg = datasets.Dataset.from_file(os.path.join(data_dir, pathes_config.vg_arrow))
    arrow_dict = {"coco": coco, "vg": vg}

    img_dirs = {
        "coco": os.path.join(data_dir, pathes_config.coco_imgs),
        "vg": os.path.join(data_dir, pathes_config.vg_imgs),
    }

    if dataset_config.split in ("pretrain", "train", "finetune"):
        files = get_file_path(data_dir, pathes_config.temp_lxmert_train)
    elif dataset_config.split in ("eval", "evaluation", "validation", "val"):
        files = get_file_path(data_dir, pathes_config.temp_lxmert_eval)
    else:
        files = get_file_path(data_dir, pathes_config.temp_lxmert_test)

    labels = os.path.join(data_dir, pathes_config.temp_lxmert_answers)
    label_data = json.load(open(labels))
    label_set = set([process_answer_default(ans["ans"]) for ans in label_data])

    return files, img_dirs, arrow_dict, label_set


if __name__ == "__main__":
    fp = "/playpen1/home/avmendoz/data/coco/test2017"
    ex = next(iter(map(lambda x: os.path.join(fp, x), os.listdir(fp))))
    ex = img_to_tensor(ex)[0]
    print(cv2.imwrite("./test.jpg", ex.permute(1, 2, 0).numpy().astype(np.uint8)))
