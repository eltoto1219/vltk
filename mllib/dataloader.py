import os
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import LxmertTokenizer

from .processors import load_temp_gqa, load_temp_lxmert
from .utils import get_duration

NAME2DATASET = {"temp_lxmert": load_temp_lxmert, "gqa": load_temp_gqa}


# probably a better collate fucntion right here
def collate_v3(
    columns: List[Dict[str, torch.Tensor]], pad: bool = True
) -> Dict[str, torch.Tensor]:
    batch = OrderedDict()
    keys = deepcopy(list(columns[0].keys()))
    for k in keys:
        if k == "raw_imgs":
            sizes = map(lambda x: x.get("raw_imgs").shape[-2:], columns)
            same_size = 1 if len(set(sizes)) == 1 else 0
            if same_size:
                batch[k] = torch.stack([i.pop(k) for i in columns if i is not None])
            else:
                max_h = max(sizes, key=lambda x: x[0])[0]
                max_w = max(sizes, key=lambda x: x[1])[1]
                batch["raw_sizes"] = torch.tensor(list(sizes))
                batch[k] = []
                for i in columns:
                    if i is None:
                        continue
                    feats_i = i.pop(k)
                    (h, w) = feats_i.shape[-2:]
                    feats_i = F.pad(feats_i, [0, max_w - w, 0, max_h - h], value=0)
                    batch[k].append(feats_i)
                batch[k] = torch.stack(batch[k])
        else:
            batch[k] = torch.stack([i.pop(k) for i in columns if i is not None])
    return batch


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

        self.set_blank_attrs()
        # props that we set ourself
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.pathes_config = pathes_config
        self.global_config = global_config
        self.train_config = train_config
        self.split = dataset_config.split if split is None else split
        self.max_objs = self.dataset_config.max_objects
        self.img_format = self.dataset_config.img_format
        self.tokenizer = LxmertTokenizer.from_pretrained(
            "unc-nlp/lxmert-base-uncased", never_split=self.never_split
        )
        self.mask_id = self.tokenizer.mask_token_id
        self.ignore_id = self.dataset_config.ignore_id
        self.pad_id = self.tokenizer.pad_token_id
        self.do_sentence_matching = self.train_config.task_matched
        self.do_language_masking = self.train_config.task_mask_lm

        # props that we load in (in this order)
        text_data, path_dict, arrow_dict, labels = self.set_dataset(dataset_name)
        self.text_data = text_data
        self.path_dict = path_dict
        self.arrow_dict = arrow_dict
        self.labels = labels
        self.set_tokenizer_args()
        self.set_id2imgidx()
        self.data_checks()

    @property
    def special_token_dict(self):
        return {
            "unk": self.tokenizer.unk_token,
            "sep": self.tokenizer.sep_token,
            "pad": self.tokenizer.pad_token,
            "cls": self.tokenizer.cls_token,
            "mask": self.tokenizer.mask_token,
        }

    def set_tokenizer_args(self):
        self.tokenizer_args = {
            "padding": "max_length",
            "max_length": self.dataset_config.sent_length,
            "truncation": self.dataset_config.truncate_sentence,
            "return_token_type_ids": self.dataset_config.return_token_type_ids,
            "return_attention_mask": self.dataset_config.return_attention_mask,
            "add_special_tokens": self.dataset_config.add_special_tokens,
            "return_tensors": self.dataset_config.return_tensors,
        }

    def set_dataset(self, dataset_name):
        text_data, path_dict, arrow_dict, labels = NAME2DATASET[dataset_name](
            self.pathes_config, self.global_config, self.dataset_config, self.split
        )
        self.num_labels = len(labels) if labels is not None else self.num_labels

        return text_data, path_dict, arrow_dict, labels

    def set_blank_attrs(self):
        self.id2imgidx = defaultdict(dict)
        self.arrow_dict = defaultdict(dict)
        self.text_files = []
        self.text_data = []
        self.num_labels = 0
        self.label2lid = {}
        self.never_split = set()
        self.labels = None
        self.tokenizer_args = {}

    def set_id2imgidx(self):
        if self.arrow_dict is not None:
            for dset in self.arrow_dict:
                self.id2imgidx[dset] = {
                    str(k): i for i, k in enumerate(self.arrow_dict[dset]["img_id"])
                }

    # later lets move this into that mapping fucnction for custom stuff

    def data_checks(self):
        assert self.num_labels <= len(
            self.labels
        ), f"{self.num_labels} {len(self.labels)}"
        print("\nnum of examples:", len(self.text_data))
        print("num labels:", self.num_labels, "\n")
        arrow_img_ids = set()
        unfound = set()
        num_unfound = 0
        if self.arrow_dict is not None:
            for x in self.id2imgidx:
                for k in self.id2imgidx[x].keys():
                    assert isinstance(k, str)
                    arrow_img_ids.add(k)
            assert len(arrow_img_ids) == sum(
                [len(self.id2imgidx[x]) for x in self.id2imgidx]
            ), len(arrow_img_ids)
            for x in self.text_data:
                text_img_id = self.clean_imgid(x["img_id"])
                assert isinstance(text_img_id, str)
                if text_img_id not in arrow_img_ids:
                    unfound.add(text_img_id)
                    num_unfound += 1

            p_unfound = num_unfound / len(self.text_data) * 100
            print(f"text-img entries skipped: {num_unfound}")
            print(f"img_ids not found: {sorted(unfound)}")
            if p_unfound == float(100):
                raise Exception(
                    f"100 {'%'} of img_ids in text data do not match img_ids in arrow dataset"
                    f"\n{self.arrow_dict}"
                )
            else:
                print(f"removing {p_unfound}% of {self.split} data")
                self.text_data = [
                    x
                    for x in self.text_data
                    if self.clean_imgid(x["img_id"]) not in unfound
                ]
                print(f"new num of text-img entries: {len(self.text_data)}")
            del arrow_img_ids
            del unfound
            del num_unfound

    def clean_imgid(self, img_id):
        return str(img_id).replace(" ", "")

    def get_img(self, dset, img_id):
        arrow_data = {}
        img_data = {}
        if self.arrow_dict is not None:
            img_idx = self.id2imgidx[dset].get(img_id, None)
            if img_idx is None and dset == "vg" and "coco" in self.id2imgidx:
                dset = "coco"
                img_idx = self.id2imgidx["coco"].get(img_id)
                if img_idx is None:
                    raise Exception(dset, img_idx, img_id)
            elif img_idx is None:
                raise Exception(dset, img_idx, img_id)
            arrow_data = self.arrow_dict[dset][img_idx]
            assert str(img_id) == str(
                arrow_data["img_id"]
            ), f"ids {img_id} != {arrow_data['img_id']}"
            arrow_data.pop("img_id")
            for k in arrow_data:
                # we discard the img_id here but maybe we can get it back later?
                if isinstance(arrow_data[k], np.ndarray):
                    arrow_data[k] = torch.from_numpy(arrow_data[k].copy())
                elif isinstance(arrow_data[k], torch.Tensor):
                    arrow_data[k] = arrow_data[k].clone()
            roi_features = arrow_data.get("roi_features", False)
            if roi_features:
                arrow_data["roi_features"] = roi_features[: self.max_objs]
        if self.dataset_config.use_raw_imgs:
            file = os.path.join(self.path_dict[dset], img_id + ".pt")
            img = torch.load(file).float()
            img_data["raw_imgs"] = img
        img_data = {**img_data, **arrow_data}
        assert img_data, "empty"
        return img_data

    def __len__(self):
        return len(self.text_data)

    @torch.no_grad()
    def __getitem__(self, i):
        entry = self.text_data[i]
        img_id = self.clean_imgid(entry.get("img_id"))
        dset = entry.get("dset")
        img_data = self.get_img(dset, img_id)
        if img_data is None:
            return {}
        # if self.do_sentence_matching:
        #     entry = self.matched_sentence_modeling(entry)
        #     img_data["is_matched"] = entry.get("is_matched")

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

        # if self.do_language_masking:
        #     input_ids = img_data.get("input_ids")
        #     masked_inds, masked_sequence = self.masked_language_modeling(input_ids)
        #     img_data["input_ids"] = masked_sequence
        #     img_data["masked_inds"] = masked_inds

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
        if split not in ("pretrain", "train", "finetune"):
            num_workers = 0
            batch_size = train_config.eval_batch_size
        drop_last = loader_config.drop_last
        pin_memory = loader_config.pin_memory
        super().__init__(
            dataset=BaseDataset(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                pathes_config=pathes_config,
                global_config=global_config,
                train_config=train_config,
                split=split,
            ),
            collate_fn=collate_v3,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        if shuffle:
            self.dataset.set_id2imgidx()

    @staticmethod
    def toCuda(batch):
        for k in batch:
            v = batch.get(k)
            batch[k] = v.cuda()
        return batch
