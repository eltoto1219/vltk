from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import LxmertTokenizer

# from mllib.dataset.gqa import load_temp_gqa
from mllib.decorators import get_duration
from mllib.maps import dirs
from mllib.processing import Image
from mllib.utils import CollatedSets

_textsets = dirs.Textsets()
_imagesets = dirs.Imagesets()


def collate(
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


class UniversalLoader(DataLoader):
    def __init__(self, names, config, split=None, textsetdict=None, imagesetdict=None):
        if split is not None and split in config.eval_aliases.union(
            config.valid_aliases
        ):
            num_workers = 0
        else:
            num_workers = config.num_workers
        shuffle = config.shuffle
        split = config.split if split is None else split
        shuffle = shuffle if (split in ("pretrain", "train")) else 0
        if config.dryrun:
            drop_last = False
        else:
            drop_last = config.drop_last
        pin_memory = config.pin_memory
        # init dataset
        dataset = UniversalDataset(
            names=names,
            config=config,
            split=split,
            textsetdict=textsetdict,
            imagesetdict=imagesetdict,
        )
        # init loader
        super().__init__(
            dataset=dataset,
            collate_fn=lambda x: collate(
                x, pad=config.pad_collate, img_first=config.img_first
            ),
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
            batch_size=dataset.batch_size,
        )

    @staticmethod
    def toCuda(batch, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        elif isinstance(device, int):
            if device == -1:
                device = "cpu"
            else:
                device = f"cuda:{device}"
        for k in batch:
            v = batch.get(k)
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                batch[k] = [i.to(device) for i in v]


class UniversalDataset(Dataset):
    def __init__(self, names, config, split=None, textsetdict=None, imagesetdict=None):
        self.config = config
        self.names = names
        self.split = split
        self.splits = []
        self.imagesets = []
        self.textsets = []
        if textsetdict is None:
            self.textsetdict = defaultdict(dict)
        else:
            self.textsetdict = textsetdict
        if imagesetdict is None:
            self.imagesetdict = defaultdict(dict)
        else:
            self.imagesetdict = imagesetdict
        if isinstance(names, str):
            names = [names]
        for name in names:
            textsets = _textsets.get(name).from_config(config, split=split)
            for split, textset in textsets.items():
                self.splits.append(split)
                self.textsets.append(textset)
                self.textsetdict[name][split] = textset
                imageset_name = textset.imageset
                if (
                    textset.name not in self.imagesetdict
                    and split not in self.imagesetdict[textset.name]
                ):
                    path_dict = textset.locations(
                        config,
                        split=split,
                        imageset=imageset_name,
                        textset=textset.name,
                    )
                    imgset_path = path_dict["arrow"]
                    assert len(imgset_path) == 1
                    from_extractor = config.from_extractor
                    imgset_path = imgset_path[0]
                    imageset = _imagesets.get(from_extractor).from_file(
                        imgset_path, name=textset.imageset
                    )
                    self.imagesets.append(imageset)
                    self.imagesetdict[textset.name][split] = imageset

        self.datatsets = CollatedSets(*self.textsets)
        self.img2textset = {}
        for ts_name, ts_splits in self.textsetdict.items():
            for split_name, ts in self.textsetdict[ts_name].items():
                for img in ts.uniq_imgs:
                    self.img2textset[img] = (ts_name, split_name)

        self.uniq_imgs = list(self.img2textset.keys())

    def __len__(self):
        if self.config.img_first:
            return len(self.uniq_imgs)
        else:
            return len(self.datasets)

    def _init_tokenizer_args(self):
        self._tokenizer_args = {
            "padding": "max_length",
            "max_length": self.config.sent_length,
            "truncation": self.config.truncate_sentence,
            "return_token_type_ids": self.config.return_token_type_ids,
            "return_attention_mask": self.config.return_attention_mask,
            "add_special_tokens": self.config.add_special_tokens,
            "return_tensors": self.config.return_tensors,
        }

    @property
    def special_token_dict(self):
        return {
            "unk": self.tokenizer.unk_token,
            "sep": self.tokenizer.sep_token,
            "pad": self.tokenizer.pad_token,
            "cls": self.tokenizer.cls_token,
            "mask": self.tokenizer.mask_token,
        }

    def __getitem__(self, i):
        if self.config.img_first:
            img_id = self.uniq_imgs[i]
            ts_name, ts_split = self.img2textset[img_id]
            img_text_dict = self.textsetdict[ts_name][ts_split].get_from_img(img_id)
            try:
                img_info_dict = self.imagesetdict[ts_name][ts_split].get(img_id)
            except Exception:
                raise Exception(self.imagesetdict[ts_name][ts_split].get_imgids)
            entry = {**img_text_dict, **img_info_dict}
            raise Exception(entry)
            return entry
        else:
            text_info = self.datasets[i]
            img_id = text_info["img_id"]
            ts_name, ts_split = self.img2textset[img_id]
            img_info_dict = self.imageset_dict[ts_name][ts_split].get(img_id)
            entry = {**text_info, **img_info_dict}
            raise Exception(entry)
            return entry

    @property
    def batch_size(self):
        if len(set(self.splits).intersection(self.config.train_aliases)) == 0:
            return self.config.eval_batch_size
        else:
            return self.config.train_batch_size

    @staticmethod
    def transpose_img2txt(batch, img_keys, device=None):
        n_sents_per_img = [len(i) for i in batch["input_ids"]]
        for img_key in img_keys:
            assert img_key in batch, f"{img_key} not in {list(batch.keys())}"
            imgs = torch.cat(
                [
                    i.unsqueeze(0).repeat((n,) + tuple([1] * (len(i.shape))))
                    for i, n in zip(batch.pop(img_key), n_sents_per_img)
                ],
                dim=0,
            )
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
