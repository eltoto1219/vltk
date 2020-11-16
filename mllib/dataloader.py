import functools
import json
import os
import timeit
from collections import OrderedDict
from test import full_img

import datasets
import numpy as np
import torch
from pynvml.smi import nvidia_smi
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import LxmertTokenizer
torch.manual_seed(1)


"""
Links to the aux data
https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt
https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt
"""


DEV = (
    sorted(
        [
            (i, d["fb_memory_usage"]["free"])
            for i, d in enumerate(
                nvidia_smi.getInstance().DeviceQuery("memory.free")["gpu"]
            )
        ],
        key=lambda x: x[0],
    )[-1][0]
    if torch.cuda.is_available()
    else -1
)

EXTRAS = ["no", "race", "religion", "sexual_attraction", "gender", "gender_identity",
"disability", "nationality", "immigration", "socioeconomic"]

def dump_split_ids(ids):
    ids = list(ids)
    total = len(ids)
    split = int(total/2)
    new1 = []
    new2 = []
    for i in range(split):
        new1.append(ids.pop())
    for i in range(split):
        new2.append(ids.pop())
    open("fhalf.json", "w").write(",".join(new1))
    open("shalf.json", "w").write(",".join(new2))


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


class MMF(Dataset):
    @get_duration
    def __init__(self, text_file, arrow_file, sent_length=20, max_objs=36, **kwargs):
        super().__init__()

        self.percent = kwargs.pop("percent", 1.0)
        self.arrow_dataset = datasets.Dataset.from_file(arrow_file)
        self.refresh_idx2id
        self.max_objs = max_objs
        self.arrow_dataset.set_format(type="numpy", output_all_columns=True)
        self.text_file = text_file
        text_data = [self.mmf_txt_format(json.loads(i)) for i in open(text_file)]
        self.text_data = text_data[:int(len(text_data) * self.percent)]
        self.tokenizer_args = {
            "padding": kwargs.pop("padding", "max_length"),
            "max_length": kwargs.pop("max_length", sent_length),
            "truncation": kwargs.pop("trunctation", True),
            "return_token_type_ids": kwargs.pop("return_token_type_ids", True),
            "return_attention_mask": kwargs.pop("return_attention_mask", True),
            "add_special_tokens": kwargs.pop("add_special_tokens", True),
            "return_tensors": kwargs.pop("return_tensors", "pt"),
        }
        self.never_split = set()
        attrs = kwargs.pop("attrs", False)
        self.attrs = [l.replace("\n", "") for l in open(attrs)] if attrs else None
        if self.attrs is not None:
            self.never_split.union(attrs)
        objs = kwargs.pop("objs", False)
        self.objs = [l.replace("\n", "") for l in open(objs)] if objs else None
        if self.objs is not None:
            self.never_split.union(objs)
        self.img_dir = kwargs.pop("img_dir", None)
        self.extract_text = kwargs.pop("extract_text", False)
        img_format = kwargs.pop("img_format", None)
        self.img_format = img_format if img_format is not None else "jpg"
        self.add_aux_data =  kwargs.pop("add_aux_data", False)

        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained(
            "unc-nlp/lxmert-base-uncased",
            never_split=self.never_split
        )
        extras = kwargs.pop("extras", False)
        self.extras = {int(k):v for k, v in json.load(open(extras)).items()} if extras else None
        if extras:
            self.text_data = [d for d in self.text_data if d[-1] in self.extras]
    def mmf_txt_format(self, entry: dict):
        return (int(entry["label"]), entry["text"], int(entry["id"]))

    def mmf_imgid_format(self, filename: str):
        return int(filename.split(".")[0])

    @torch.no_grad()
    def mmf_extras(self, iid: int):
        assert self.extras is not None
        entry = self.extras[iid]
        classification = EXTRAS.index(entry["classification"])
        truth = torch.Tensor([classification]).long()
        return truth

    def process_aux(self, label_inds: torch.Tensor, aux_type: str):
        aux = getattr(self, aux_type, None)
        assert aux is not None
        aargs = {
            "return_tensors": "pt",
            "padding": "max_length",
            "return_attention_mask": True,
            "add_special_tokens": False,
            "truncation": True,
            "max_length": self.max_objs
        }
        aux_batch = " ".join([
            aux[int(i)].split(",")[0].replace(" ", "")
            for i in label_inds.int().tolist()
        ])
        aux_batch = self.lxmert_tokenizer(aux_batch, **aargs)
        aux_batch.input_ids = aux_batch.input_ids.permute(1,0)
        aux_batch.token_type_ids = aux_batch.token_type_ids.permute(1,0)
        aux_batch.attention_mask= aux_batch.attention_mask.permute(1,0)
        return aux_batch

    def view_aux_data_as_text(self, lxmert_label_inds):
        """
        this is decieving becuase huggingface uses the wordpiece tokenizer which splits
        compound words (ex: "toothbrush" to "tooth" "##brush") and then truncates
        anything over 36 objects, so the objtects/attributes will not be perfectly
        aliged
        """
        # note, the label inds have already been converted to lxmert at this point
        if lxmert_label_inds.ndim == 1:
            return self.lxmert_tokenizer.convert_ids_to_tokens(
                lxmert_label_inds.int().tolist(), skip_special_tokens=True
            )
        else:
            return [
                self.lxmert_tokenizer.convert_ids_to_tokens(l, skip_special_tokens=True)
                for l in lxmert_label_inds.int().tolist()
            ]

    def __len__(self):
        return len(self.text_data)

    @property
    def refresh_idx2id(self):
        self.id2idx = {k: i for i, k in enumerate(self.arrow_dataset["img_id"])}

    def _get_img_path(self, img_id):
        img_id = str(int(img_id)) + f".{self.img_format}"
        assert self.img_dir is not None
        for i in range(12):
            tmp_id = "0" * i + img_id
            path = os.path.realpath(os.path.join(self.img_dir, tmp_id))
            if os.path.isfile(path):
                return path
        raise Exception(f"no img found for id {path}, try different img_format?")

    def _extract_text(self, img_id):
            ignore = []
            path = self._get_img_path(img_id)
            results = full_img(path=path, ind=None)
            for r in results:
                ignore.append([r[1][1], r[1][3]])
            return ignore

    @torch.no_grad()
    def __getitem__(self, i):
        label, text, img_id = self.text_data[i]
        img_idx = self.id2idx[img_id]
        img_data = self.arrow_dataset[img_idx]
        assert img_id == img_data["img_id"], f"ids {img_id} != {img_data['img_id']}"
        for k in img_data:
            if isinstance(img_data[k], np.ndarray):
                img_data[k] = torch.from_numpy(img_data[k].copy())
            elif isinstance(img_data[k], torch.Tensor):
                img_data[k] = img_data[k].clone()
        if self.extract_text:
            img_data["ignore"] = torch.Tensor(self._extract_text(img_id))
        inputs = self.lxmert_tokenizer(text, **self.tokenizer_args)
        img_data["input_ids"] = inputs.input_ids.squeeze(0)
        img_data["attention_mask"] = inputs.attention_mask.squeeze(0)
        img_data["token_type_ids"] = inputs.token_type_ids.squeeze(0)
        img_data["label"] = torch.tensor(label)
        img_data["roi_features"] = img_data["roi_features"][:self.max_objs]
        if self.extras:
            img_data["extras"] = self.mmf_extras(img_id)
        # returns the attr/obj ids except the tokenized inds are from the lxmert vocab
        if self.attrs is not None and "attr_ids" in img_data:
            img_data["attr_probs"] =  img_data["attr_probs"][:self.max_objs]
            attr_dict = self.process_aux(img_data.get("attr_ids"), aux_type="attrs")
            img_data["attr_input_ids"] = attr_dict.input_ids
            img_data["attr_input_mask"] = attr_dict.attention_mask
        if self.objs is not None and "obj_ids" in img_data:
            img_data["obj_probs"] =  img_data["obj_probs"][:self.max_objs]
            obj_dict = self.process_aux(img_data.get("obj_ids"), aux_type="objs")
            img_data["obj_input_ids"] = obj_dict.input_ids
            img_data["obj_input_mask"] = obj_dict.attention_mask
        return img_data


class MMFLoader(DataLoader):
    def __init__(self, text_file, arrow_file, **kwargs):
        shuffle = kwargs.get("shuffle", True)
        shuffle = 1 if (shuffle and "train" in text_file) else 0
        num_workers = kwargs.pop("num_workers", 4)
        drop_last = kwargs.pop("drop_last", False)
        pin_memory = kwargs.pop("pin_memory", True)
        batch_size = kwargs.pop("batch_size", 1)
        return_tensor = kwargs.pop("collate_pt", True)
        super().__init__(
            dataset=MMF(text_file, arrow_file, **kwargs),
            collate_fn=collate_tensor if return_tensor else collate_list,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
            batch_size=batch_size,
        )
        if shuffle:
            self.dataset.refresh_idx2id

    @staticmethod
    def toCuda(batch):
        for k in batch:
            v = batch.get(k)
            batch[k] = v.cuda()
        return batch


if __name__ == "__main__":
    loader = MMFLoader(
        "train.jsonl",
        "arrow/mmf.arrow",
        max_objs=36,
        sent_length=14,
        num_workers=0,
        batch_size=1,
        objs="objects.txt",
        attrs="attributes.txt",
        percent=1.0,
        add_aux_data=True,
        extras='./darryl.json'
    )

    ids = set()
    for x in loader:
        i = str(x["img_id"])
        ids.add(i)
    dump_split_ids(ids)


