import json
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import LxmertTokenizer

from .mapping import NAME2DATASET, NAME2PROCESSOR
from .utils import get_duration, get_file_path


def collate_list(columns):
    batch = OrderedDict()
    for x in map(lambda x: iter(x.items()), columns):
        for k, v in x:
            if batch.get(k) is None:
                batch[k] = [v]
            else:
                batch[k].append(v)
    return batch


# probably a better collate fucntion right here
def collate_v3(
    columns: List[Dict[str, torch.Tensor]], pad: bool = True
) -> Dict[str, torch.Tensor]:
    batch = OrderedDict()
    keys = deepcopy(list(columns[0].keys()))
    for k in keys:
        if k == "img_features":
            if pad:
                max_h = max([int(i.get("sizes")[0]) for i in columns])
                max_w = max([int(i.get("sizes")[1]) for i in columns])
            batch[k] = []
            for i in iter(map(lambda x: x, columns)):
                if i is None:
                    continue
                feats_i = i.pop(k)
                if pad:
                    (h, w) = feats_i.shape[-2:]
                    feats_i = F.pad(feats_i, [0, max_w - w, 0, max_h - h], value=0)
                batch[k].append(feats_i)

        else:
            if k != "sizes":
                batch[k] = [i.pop(k) for i in columns if i is not None]
            else:
                batch[k] = [i.get(k) for i in columns if i is not None]
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

        self.set_blank_attrs()
        # props that we set ourself
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.pathes_config = pathes_config
        self.global_config = global_config
        self.train_config = train_config
        self.split = dataset_config.split if split is None else split
        self.img_processor = dataset_config.img_processor
        self.max_objs = self.dataset_config.max_objects
        self.tokenizer = LxmertTokenizer.from_pretrained(
            "unc-nlp/lxmert-base-uncased", never_split=self.never_split
        )
        self.mask_id = self.tokenizer.mask_token_id
        self.ignore_id = self.dataset_config.ignore_id
        self.pad_id = self.tokenizer.pad_token_id
        self.do_sentence_matching = self.train_config.task_matched
        self.do_language_masking = self.train_config.task_mask_lm

        # props that we load in
        img_dict, arrow_dict = self.set_dataset(dataset_name)
        self.set_img_processing(img_dict, arrow_dict)
        self.set_tokenizer_args()
        self.set_text_files()
        self.set_id2idx()
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
        text_files, img_dict, arrow_dict, labels = NAME2DATASET[dataset_name](
            self.pathes_config, self.global_config, self.dataset_config, self.split
        )

        self.labels = labels
        self.text_files = text_files

        return img_dict, arrow_dict

    def set_img_processing(self, img_dict=None, arrow_dict=None):
        self.img_format = self.dataset_config.img_format
        if self.dataset_config.use_arrow:
            assert arrow_dict is not None
            self.img_dict = arrow_dict
            for k in self.img_dict:
                self.img_dict[k].set_format(type="numpy", output_all_columns=True)
        else:
            assert img_dict is not None
            for dset in img_dict:
                self.img_dict[dset] = {
                    self.clean_imgid(x.split("/")[-1].split(".")[0]): x
                    for x in get_file_path(
                        img_dict[dset],
                    )
                }

    def set_blank_attrs(self):
        self.id2idx = defaultdict(dict)
        self.img_dict = defaultdict(dict)
        self.text_files = []
        self.text_data = []
        self.num_labels = 0
        self.label2lid = {}
        self.never_split = set()
        self.labels = None
        self.tokenizer_args = {}

    def set_id2idx(self):
        if self.dataset_config.use_arrow:
            for dset in self.img_dict:
                self.id2idx[dset] = {
                    str(k): i for i, k in enumerate(self.img_dict[dset]["img_id"])
                }

    # later lets move this into that mapping fucnction for custom stuff
    def set_text_files(self):
        ignored_labels = set()
        num_ignored = 0
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
                                        k = NAME2PROCESSOR["default_ans"](k)
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
            self.num_labels = len(self.label2lid)
            for text in tqdm(self.text_files):
                data_split = json.load(open(text))
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
                        l = NAME2PROCESSOR["default_ans"](l)
                        if l not in self.labels:
                            num_ignored += 1
                            ignored_labels.add(l)
                            continue
                        else:
                            entry["label"] = torch.tensor(self.label2lid[l])
                            self.text_data.append(entry)
            print(f"num ignored: {num_ignored} ignored: {ignored_labels}")

    def data_checks(self):
        assert self.num_labels <= len(
            self.labels
        ), f"{self.num_labels} {len(self.labels)}"
        print("\nnum of examples:", len(self.text_data))
        print("num labels:", self.num_labels, "\n")
        arrow_img_ids = set()
        unfound = set()
        num_unfound = 0
        if self.dataset_config.use_arrow:
            for x in self.id2idx:
                for k in self.id2idx[x].keys():
                    assert isinstance(k, str)
                    arrow_img_ids.add(k)
            assert len(arrow_img_ids) == sum(
                [len(self.id2idx[x]) for x in self.id2idx]
            ), len(arrow_img_ids)
        else:
            for x in self.img_dict:
                for k in self.img_dict[x].keys():
                    assert isinstance(k, str)
                    arrow_img_ids.add(k)

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
                f"\n{self.img_dict}"
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
        return "".join([x for x in str(img_id).lstrip("0") if x.isdigit()])

    def get_img(self, dset, img_id):
        if self.dataset_config.use_arrow:
            img_idx = self.id2idx[dset].get(img_id, None)
            if img_idx is None and dset == "vg" and "coco" in self.id2idx:
                dset = "coco"
                img_idx = self.id2idx["coco"].get(img_id)
                if img_idx is None:
                    raise Exception(dset, img_idx, img_id)
            elif img_idx is None:
                raise Exception(dset, img_idx, img_id)
            img_data = self.img_dict[dset][img_idx]
            assert str(img_id) == str(
                img_data["img_id"]
            ), f"ids {img_id} != {img_data['img_id']}"
            for k in img_data:
                if isinstance(img_data[k], np.ndarray):
                    img_data[k] = torch.from_numpy(img_data[k].copy())
                elif isinstance(img_data[k], torch.Tensor):
                    img_data[k] = img_data[k].clone()
            img_data["roi_features"] = img_data.get("roi_features")[: self.max_objs]
        else:
            img_data = {}
            fp = self.img_dict[dset][img_id]
            img_tensor, (orig_h, orig_w), (new_h, new_w) = NAME2PROCESSOR[
                self.img_processor
            ](fp)
            if img_tensor is None:
                return None
            img_data["img_features"] = img_tensor
            img_data["sizes"] = torch.Tensor([orig_h, orig_w])
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
            # collate_fn=collate_tensor if loader_config.collate_pytorch else collate_list,
            collate_fn=collate_v3,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        if shuffle:
            self.dataset.set_id2idx()

    @staticmethod
    def toCuda(batch):
        for k in batch:
            v = batch.get(k)
            batch[k] = v.cuda()
        return batch
