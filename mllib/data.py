import os
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import LxmertTokenizer

from mllib.processors import datasets, images
from mllib.utils import get_duration


# probably a better collate fucntion right here
def collate_v3(
    columns: List[Dict[str, torch.Tensor]], pad: bool = True, img_first=False
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
            if isinstance(columns[0].get(k), torch.Tensor) and not img_first:
                batch[k] = torch.stack([i.pop(k) for i in columns if i is not None])
            else:
                batch[k] = [i.pop(k) for i in columns if i is not None]
    return batch


class UniversalDataset(Dataset):
    @get_duration
    def __init__(self, dataset_name, config, split=None):
        super().__init__()

        self.set_blank_attrs()
        # props that we set ourself
        self.dataset_name = dataset_name
        self.config = config
        self.split = config.data.split if split is None else split
        self.max_objs = self.config.data.max_objects
        self.img_format = self.config.data.img_format
        self.tokenizer = LxmertTokenizer.from_pretrained(
            "unc-nlp/lxmert-base-uncased", never_split=self.never_split
        )
        self.mask_id = self.tokenizer.mask_token_id
        self.ignore_id = self.config.data.ignore_id
        self.pad_id = self.tokenizer.pad_token_id
        self.do_sentence_matching = self.config.run.task_matched
        self.do_language_masking = self.config.run.task_mask_lm

        # props that we load in (in this order)
        text_data, path_dict, arrow_dict, labels = self.set_dataset(dataset_name)
        self.text_data = tuple(text_data[
            : max(1, int(np.ceil(len(text_data) * self.config.data.percent_data)))
        ])
        self.img_ids = []
        for t in self.text_data:
            img_id = t.get("img_id")
            self.img_ids.append(img_id)
            self.imgid2text[img_id].append(t)
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
            "max_length": self.config.data.sent_length,
            "truncation": self.config.data.truncate_sentence,
            "return_token_type_ids": self.config.data.return_token_type_ids,
            "return_attention_mask": self.config.data.return_attention_mask,
            "add_special_tokens": self.config.data.add_special_tokens,
            "return_tensors": self.config.data.return_tensors,
        }

    def set_dataset(self, dataset_name):
        text_data, path_dict, arrow_dict, labels = datasets.NAME2DATASET[dataset_name](
            self.config, self.split
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
        self.imgid2text = defaultdict(list)
        self.img_ids = []

    def get_img_first_entry(self, i):
        img_id = self.img_ids[i]
        text_data = self.imgid2text[img_id]
        dset = text_data[0]["dset"]
        return img_id, text_data, dset

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
            img_ids = arrow_data.pop("img_id")
            for k in arrow_data:
                # we discard the img_id here but maybe we can get it back later?
                if isinstance(arrow_data[k], np.ndarray):
                    arrow_data[k] = torch.from_numpy(arrow_data[k].copy())
                elif isinstance(arrow_data[k], torch.Tensor):
                    arrow_data[k] = arrow_data[k].clone()
            arrow_data["img_ids"] = img_ids
            roi_features = arrow_data.get("roi_features", None)
            if roi_features is not None:
                arrow_data["roi_features"] = roi_features[: self.max_objs]
        if self.config.data.use_raw_imgs:
            file = os.path.join(
                self.path_dict[dset], img_id + f".{self.config.data.img_format}"
            )
            assert os.path.isfile(file), file
            img, (ogh, ogw), scales = images.img_to_tensor(
                file,
                min_size=self.config.data.img_max_size,
                max_size=self.config.data.img_max_size,
                use_gpu=False,
                pad_value=0,
            )
            img_data["raw_imgs"] = img
        img_data = {**img_data, **arrow_data}
        assert img_data, "empty"
        return img_data

    def __len__(self):
        return len(self.text_data)

    @torch.no_grad()
    def __getitem__(self, i):
        if not self.config.data.img_first:
            # get one
            entry = self.text_data[i]
            img_id = self.clean_imgid(entry.get("img_id"))
            dset = entry.get("dset")
            entries = [entry]
        else:
            # get many
            img_id, entries, dset = self.get_img_first_entry(i)

        img_data = self.get_img(dset, img_id)
        input_ids = []
        attention_mask = []
        token_type_ids = []
        labels = []
        # if self.do_sentence_matching:
        #     entry = self.matched_sentence_modeling(entry)
        #     img_data["is_matched"] = entry.get("is_matched")
        if img_data is None:
            # this means that somehow while loading the image, it returned None
            return {}
        inputs = []
        labels = []
        for entry in entries:
            inputs.append(self.tokenizer(entry.get("text"), **self.tokenizer_args))
            labels.append(entry.get("label"))
        input_ids = [i.input_ids.squeeze(0) for i in inputs]
        attention_mask = [i.attention_mask.squeeze(0) for i in inputs]
        token_type_ids = [i.token_type_ids.squeeze(0) for i in inputs]
        img_data["input_ids"] = (
            input_ids[0] if not self.config.data.img_first else torch.stack(input_ids)
        )
        img_data["attention_mask"] = (
            attention_mask[0]
            if not self.config.data.img_first
            else torch.stack(attention_mask)
        )
        img_data["token_type_ids"] = (
            token_type_ids[0]
            if not self.config.data.img_first
            else torch.stack(token_type_ids)
        )
        img_data["labels"] = labels[0] if not self.config.data.img_first else torch.stack(labels)
        # if self.do_language_masking:
        #     input_ids = img_data.get("input_ids")
        #     masked_inds, masked_sequence = self.masked_language_modeling(input_ids)
        #     img_data["input_ids"] = masked_sequence
        #     img_data["masked_inds"] = masked_inds

        return img_data

    @staticmethod
    def transpose_img2txt(batch, img_keys, device=None):
        n_sents_per_img = [len(i) for i in batch["input_ids"]]
        for img_key in img_keys:
            if img_key in batch:
                if batch[img_key].size(0) == 1:
                    imgs = batch[img_key].unsqueeze(0)
                else:
                    imgs = torch.cat([
                        i.unsqueeze(0).repeat((n,) + tuple([1] * (len(i.shape))))
                        for i, n in zip(batch.pop(img_key), n_sents_per_img)
                    ], dim=0)
                batch[img_key] = imgs
                if device is not None:
                    batch[img_key] = batch[img_key].to(device)
        for k in batch:
            if k not in img_keys:
                if isinstance(batch[k][0], torch.Tensor):
                    batch[k] = torch.cat([i for i in batch[k]], dim=0)
                    if device is not None:
                        batch[k] = batch[k].to(device)
                elif isinstance(batch[k][0], str):
                    new_v = []
                    for i, n in zip(batch[k], n_sents_per_img):
                        new_v.extend(i * n)
                    batch[k] = new_v


class UniversalLoader(DataLoader):
    def __init__(self, dataset_name, config, split=None):
        if split is not None and split not in ("pretrain", "train", "finetune"):
            if hasattr(config, "eval"):
                batch_size = config.eval.batch_size
            else:
                batch_size = config.run.batch_size
            num_workers = 0
        else:
            batch_size = config.run.batch_size
            num_workers = config.data.num_workers
        shuffle = config.data.shuffle
        split = config.data.split if split is None else split
        shuffle = shuffle if (split in ("pretrain", "train")) else 0
        if config.dryrun:
            drop_last = False
        else:
            drop_last = config.data.drop_last
        pin_memory = config.data.pin_memory
        super().__init__(
            dataset=UniversalDataset(
                dataset_name=dataset_name,
                config=config,
                split=split,
            ),
            collate_fn=lambda x: collate_v3(x, pad=config.data.pad_collate, img_first=config.data.img_first),
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        if shuffle:
            self.dataset.set_id2imgidx()

    @staticmethod
    def toCuda(batch, device=None):
        if isinstance(device, int):
            if device == -1:
                device = "cpu"
            else:
                device = f"cuda:{device}"
        for k in batch:
            v = batch.get(k)
            if isinstance(v, torch.Tensor):
                if device is None:
                    if torch.cuda.is_available():
                        batch[k] = v.cuda()
                    else:
                        batch[k] = v.cpu()
                else:
                    batch[k] = v.to(device)
