import functools
import json
import os
import random
import timeit
from collections import OrderedDict, defaultdict

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import LxmertTokenizer

"""
Links to the aux data
https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt
https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt
"""

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


def get_file_path(data_dir, relative_path):
    files = []
    fp = os.path.join(data_dir, relative_path)
    if os.path.isdir(fp):
        for x in os.listdir(fp):
            files.append(os.path.join(fp, x))
    elif os.path.isfile(fp):
        files.append(fp)
    else:
        raise Exception(fp)

    return files


def convert_answer(ans):
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


def load_images(img_dirs):
    imgid2file = {}
    return imgid2file


def load_temp_gqa(
    pathes_config,
    global_config,
    dataset_config,
    split,
):
    data_dir = global_config.data_dir
    vg = datasets.Dataset.from_file(os.path.join(data_dir, pathes_config.vg_arrow))
    arrow_dict = {"gqa": vg}

    img_dirs = [
        os.path.join(data_dir, pathes_config.vg_imgs),
    ]

    if split in ("pretrain", "train", "finetune"):
        files = get_file_path(data_dir, pathes_config.gqa_train)
    elif split in ("eval", "evaluation", "validation", "val"):
        files = get_file_path(data_dir, pathes_config.gqa_val)
    else:
        files = get_file_path(data_dir, pathes_config.gqa_test)

    labels = os.path.join(data_dir, pathes_config.gqa_answers)
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

    img_dirs = [
        os.path.join(data_dir, pathes_config.coco_imgs),
        os.path.join(data_dir, pathes_config.vg_imgs),
    ]

    if dataset_config.split in ("pretrain", "train", "finetune"):
        files = get_file_path(data_dir, pathes_config.temp_lxmert_train)
    elif dataset_config.split in ("eval", "evaluation", "validation", "val"):
        files = get_file_path(data_dir, pathes_config.temp_lxmert_eval)
    else:
        files = get_file_path(data_dir, pathes_config.temp_lxmert_test)

    labels = os.path.join(data_dir, pathes_config.temp_lxmert_answers)
    label_data = json.load(open(labels))
    label_set = set([convert_answer(ans["ans"]) for ans in label_data])

    return files, img_dirs, arrow_dict, label_set


DATASET2DATA = {"temp_lxmert": load_temp_lxmert, "gqa": load_temp_gqa}


def get_duration(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        starttime = timeit.default_timer()
        output = func(*args, **kwargs)
        print(f"exec: {func.__name__} in {timeit.default_timer() - starttime:.3f} s")
        return output

    return wrapper


def collate_list(columns):
    batch = OrderedDict()
    for x in map(lambda x: iter(x.items()), columns):
        for k, v in x:
            if batch.get(k) is None:
                batch[k] = [v]
            else:
                batch[k].append(v)
    return batch


def collate_tensor(columns):
    batch = OrderedDict()
    for x in map(lambda x: iter(x.items()), columns):
        for k, v in x:
            if batch.get(k) is None:
                batch[k] = [v]
            else:
                batch[k].append(v)
    return {k: torch.stack(v) for k, v in batch.items()}


class BaseDataset(Dataset):
    @get_duration
    def __init__(
        self,
        dataset_name,
        dataset_config,
        pathes_config,
        global_config,
        train_config,
        split=None,
    ):
        super().__init__()
        # setup stuff
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.pathes_config = pathes_config
        self.global_config = global_config
        self.train_config = train_config
        self.split = dataset_config.split if split is None else split

        files, img_dirs, arrow_dict, labels = DATASET2DATA[dataset_name](
            pathes_config, global_config, dataset_config, self.split
        )
        print("loaded_files_and_image_dirs")
        self.id2idx = defaultdict(dict)
        self.labels = labels
        self.text_files = files
        self.text_data = []
        if dataset_config.use_arrow:
            self.arrow_dict = arrow_dict
            self.img_files = None
        else:
            self.img_files = load_images(img_dirs)
            self.arrow_dataset = None

        self.img_format = self.dataset_config.img_format

        # handles specific datasettings
        self.max_objs = self.dataset_config.max_objects
        for k in self.arrow_dict:
            self.arrow_dict[k].set_format(type="numpy", output_all_columns=True)
        self.tokenizer_args = {
            "padding": "max_length",
            "max_length": self.dataset_config.sent_length,
            "truncation": self.dataset_config.truncate_sentence,
            "return_token_type_ids": self.dataset_config.return_token_type_ids,
            "return_attention_mask": self.dataset_config.return_attention_mask,
            "add_special_tokens": self.dataset_config.add_special_tokens,
            "return_tensors": self.dataset_config.return_tensors,
        }
        self.never_split = set()
        self.tokenizer = LxmertTokenizer.from_pretrained(
            "unc-nlp/lxmert-base-uncased", never_split=self.never_split
        )

        self.mask_id = self.tokenizer.mask_token_id
        self.ignore_id = self.dataset_config.ignore_id
        self.pad_id = self.tokenizer.pad_token_id
        self.do_sentence_matching = self.train_config.msm
        self.do_language_masking = self.train_config.mlm
        self.label2lid = {}
        self.num_labels = 0  # it self counts
        self.load_text_files()
        self.refresh_id2idx()
        assert self.num_labels <= len(
            self.labels
        ), f"{self.num_labels} {len(self.labels)}"
        assert len(self.label2lid) == self.num_labels

    @property
    def special_ids(self):
        return set(
            [
                int(self.tokenizer.unk_token_id),
                int(self.tokenizer.sep_token_id),
                int(self.tokenizer.pad_token_id),
                int(self.tokenizer.cls_token_id),
                int(self.tokenizer.mask_token_id),
            ]
        )

    def random_ind(self):
        token = random.choice(list(self.tokenizer.vocab.items()))
        return int(token[-1])

    def refresh_id2idx(self):
        for dset in self.arrow_dict:
            self.id2idx[dset] = {
                str(k): i for i, k in enumerate(self.arrow_dict[dset]["img_id"])
            }

    # later lets move this into that mapping fucnction for custom stuff
    def load_text_files(self):
        name2dset = {
            "mscoco": "coco",
            "vg": "vg",
            "visual7w": "vg",
            "gqa": "vg",
            "vqa": "coco",
        }
        if self.dataset_name == "temp_lxmert":
            print("loading text data")
            for text in tqdm(self.text_files):
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
                                        k = convert_answer(k)
                                        if k in self.labels or k in self.label2lid:
                                            in_label_set = True
                                            label = k
                                            break
                                    if in_label_set:
                                        if label not in self.label2lid:
                                            lid = self.num_labels
                                            self.label2lid[label] = lid
                                            self.num_labels += 1
                                        else:
                                            lid = self.label2lid[label]
                                    else:
                                        lid = self.ignore_id
                            else:
                                lid = self.ignore_id
                            entry["label"] = torch.tensor(lid)
                            self.text_data.append(entry)
        elif self.dataset_name == "gqa":
            self.label2lid = self.labels
            self.num_labels = len(self.labels)
            for text in tqdm(self.text_files):
                data_split = json.load(open(text))
                for data in data_split:
                    img_id = data["img_id"].split("_")[-1]
                    entry = {"img_id": img_id, "text": data["sent"], "dset": "gqa"}
                    try:
                        label = next(iter(data["label"].keys()))
                    except StopIteration:
                        continue
                    entry["label"] = self.labels(label)
                    self.text_data.append(entry)

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

    def __len__(self):
        return len(self.text_data)

    @torch.no_grad()
    def __getitem__(self, i):
        entry = self.text_data[i]
        img_id = entry.get("img_id").lstrip("0")
        dset = entry.get("dset")
        img_idx = self.id2idx[dset].get(img_id, None)
        if img_idx is None and dset == "vg":
            dset = "coco"
            img_idx = self.id2idx["coco"].get(img_id)
            if img_idx is None:
                print("no id")
                pass
        img_data = self.arrow_dict[dset][img_idx]
        assert str(img_id) == str(
            img_data["img_id"]
        ), f"ids {img_id} != {img_data['img_id']}"
        for k in img_data:
            if isinstance(img_data[k], np.ndarray):
                img_data[k] = torch.from_numpy(img_data[k].copy())
            elif isinstance(img_data[k], torch.Tensor):
                img_data[k] = img_data[k].clone()

        if self.do_sentence_matching:
            entry = self.matched_sentence_modeling(entry)
            img_data["is_matched"] = entry.get("is_matched")

        inputs = self.tokenizer(entry.get("text"), **self.tokenizer_args)
        img_data["input_ids"] = inputs.input_ids.squeeze(0)
        img_data["attention_mask"] = inputs.attention_mask.squeeze(0)
        img_data["token_type_ids"] = inputs.token_type_ids.squeeze(0)
        if not isinstance(entry["label"], torch.Tensor):
            if isinstance(entry["label"], int):
                entry["label"] = torch.tensor(entry["label"])
            else:
                raise Exception(f"entry is of type: {type(entry['label'])}")
        img_data["label"] = entry.get("label")
        img_data["roi_features"] = img_data.get("roi_features")[: self.max_objs]

        if self.do_language_masking:
            input_ids = img_data.get("input_ids")
            masked_inds, masked_sequence = self.masked_language_modeling(input_ids)
            img_data["input_ids"] = masked_sequence
            img_data["masked_inds"] = masked_inds

        return img_data


class BaseLoader(DataLoader):
    def __init__(
        self,
        dataset_name,
        dataset_config,
        loader_config,
        global_config,
        pathes_config,
        train_config,
        split=None,
    ):
        batch_size = train_config.train_batch_size
        shuffle = loader_config.shuffle
        split = dataset_config.split if split is None else split
        shuffle = shuffle if (split in ("pretrain", "train")) else 0
        num_workers = loader_config.num_workers
        if split == "eval":
            num_workers = 0
            batch_size = train_config.val_batch_size
        drop_last = loader_config.drop_last
        pin_memory = loader_config.pin_memory
        return_tensor = loader_config.collate_pytorch
        super().__init__(
            dataset=BaseDataset(
                dataset_name,
                dataset_config,
                pathes_config,
                global_config,
                train_config,
                split=split,
            ),
            collate_fn=collate_tensor if return_tensor else collate_list,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        if shuffle:
            self.dataset.refresh_id2idx()

    @staticmethod
    def toCuda(batch):
        for k in batch:
            v = batch.get(k)
            batch[k] = v.cuda()
        return batch
