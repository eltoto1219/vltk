# note if we do not immport a pacakage correctly in this class, no loops or exps will be present
import json
import os
import random
import resource
from collections.abc import Iterable
from copy import deepcopy
from typing import Dict, List

import numpy
import PIL.Image as Image
import torch
import torch.nn.functional as F
# disable logging from datasets
from datasets.utils.logging import set_verbosity_error
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader, Dataset

from vltk import IMAGEKEY, LABELKEY, RAWIMAGEKEY, SCOREKEY, TEXTKEY
from vltk.inspect import collect_args_to_func
# from vltk.dataset.gqa import load_temp_gqa
from vltk.processing import data as data_proc
from vltk.processing import image as image_proc

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (6144, rlimit[1]))

set_verbosity_error()

VOCABPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "libdata/bert-base-uncased-vocab.txt")
)
TOKENIZEDKEY = "encoded"
global TORCHCOLS
TORCHCOLS = set()
os.environ["TOKENIZERS_PARALLELISM"] = "False"

_data_procecessors = data_proc.Data()
_image_preprocessors = image_proc.Image()


class CollatedSets:
    def __init__(self, *args):
        self.args = args
        self.range2listpos = {}
        start = 0
        for i, a in enumerate(args):
            self.range2listpos[range(start, len(a) + start)] = i
            start += len(a)

    def get_textset_and_ind(self, x):
        if x >= len(self):
            raise IndexError(f"index {x} is out of range 0 to {len(self)}")
        for rng in self.range2listpos:
            if x in rng:
                listpos = self.range2listpos[rng]
                listind = x - rng.start
                return self.args[listpos], listind

    def __getitem__(self, x):
        if x >= len(self):
            raise IndexError(f"index {x} is out of range 0 to {len(self)}")
        for rng in self.range2listpos:
            if x in rng:
                listpos = self.range2listpos[rng]
                listind = x - rng.start
                return self.args[listpos][listind]

    def __len__(self):
        return sum(map(lambda x: len(x), self.args))

    def __iter__(self):
        return iter(map(lambda x: self[x], range(0, len(self))))


def collate(
    columns: List[Dict[str, torch.Tensor]], pad: bool = True, img_first=False
) -> Dict[str, torch.Tensor]:
    batch = {}
    columns = sorted(columns, key=lambda x: len(x), reverse=True)
    keys = deepcopy(list(columns[0].keys()))

    for k in keys:
        try:

            batch[k] = torch.stack([i.get(k) for i in columns if i is not None])
        except Exception:
            # print("THIS IS K", k, columns[0].get(k, ""), columns[0].keys())
            # print()
            batch[k] = [i.get(k, "") for i in columns if i is not None]

    return batch


class UniversalLoader(DataLoader):
    def __init__(
        self,
        names,
        config,
        label_dict,
        textsetdict,
        imagesetdict,
    ):
        splits = set()
        for v in textsetdict.values():
            splits = splits.union(set(v.keys()))
        if "train" not in splits or "pretrain" not in splits:
            num_workers = 0
        else:
            num_workers = config.num_workers
        shuffle = config.shuffle
        shuffle = shuffle if set(splits).union(config.train_aliases) else 0
        drop_last = config.drop_last
        pin_memory = config.pin_memory
        # init dataset
        dataset = UniversalDataset(
            names=names,
            config=config,
            label_dict=label_dict,
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


class UniversalDataset(Dataset):
    def __init__(self, names, config, label_dict, textsetdict, imagesetdict):
        self.config = config
        self.names = names
        self.tokenizer = BertWordPieceTokenizer(VOCABPATH, lowercase=True)
        self.tokenizer.add_special_tokens(self.special_tokens)
        self.tokenizer.enable_truncation(max_length=config.sent_length)
        self.tokenizer.enable_padding(length=config.sent_length)
        self._init_cache_batch()
        splits = set()
        for v in textsetdict.values():
            splits = splits.union(set(v.keys()))
        self.splits = splits
        self.textsets = []
        self.textsetdict = textsetdict
        self.imagesetdict = imagesetdict
        self.label_to_id = label_dict
        self.uniq_labels = set(label_dict.keys())
        textsets = []
        # map special function and create list of textsets
        for dset in self.textsetdict:
            for split in self.textsetdict[dset]:
                textset = self.textsetdict[dset][split]
                textsets.append(textset)
                self.textsetdict[dset][split] = textset
        self.datasets = CollatedSets(*textsets)
        self.img2textset = {}
        self.uniq_imgs = set()
        for ts_name, ts_splits in self.textsetdict.items():
            for split_name, ts in self.textsetdict[ts_name].items():
                self.uniq_imgs = self.uniq_imgs.union(ts.uniq_imgs)
                for img in ts.uniq_imgs:
                    self.img2textset[img] = (ts_name, split_name)
        self.uniq_imgs = list(self.uniq_imgs)
        special_ids = set([self.tokenizer.token_to_id(t) for t in self.special_tokens])
        self.special_ids = deepcopy(special_ids)
        all_ids = [i[1] for i in self.tokenizer.get_vocab().items()]
        self.all_ids = deepcopy(all_ids)

    def update_labels(self, path_or_dict):
        if isinstance(path_or_dict, str):
            path_or_dict = json.load(open(path_or_dict))
        else:
            pass
        self.label_to_id = path_or_dict
        self.uniq_labels = set(path_or_dict.keys())

    def _init_cache_batch(self):
        if self.config.img_first:
            cache_batch_path = os.path.join(
                self.config.logdir, "img_first_" + self.config.cache_batch.lstrip("/")
            )
        else:
            cache_batch_path = os.path.join(self.config.logdir, self.config.cache_batch)

        self.cache_batch_path = cache_batch_path
        self.cache_batch_exists = os.path.isfile(cache_batch_path)

    def processor_args(self):
        max_rand_sents = 1 if not self.config.img_first else 32
        return {
            "tokenizer": self.tokenizer,
            "config": self.config,
            "random_sents": [self.random_sent() for i in range(max_rand_sents)],
            "special_ids": self.special_ids,
            "label_to_id": self.label_to_id,
            "all_ids": self.all_ids,
            "n_ids": len(self.all_ids),
        }

    @staticmethod
    def text_map_function(x, proc_args):
        config = proc_args.get("config")
        tokenizer = proc_args.get("tokenizer")
        label_to_id = proc_args.get("label_to_id")
        text_processors = config.text_processors
        if text_processors is not None:
            if "matched_sentence_modeling" in text_processors:
                proc_func = _data_procecessors.get("matched_sentence_modeling")
                x = proc_func(x, **proc_args)
                TORCHCOLS.add("is_matched")

        encoded = tokenizer.encode(x.pop(TEXTKEY))
        x.pop("img_id", None)
        x["text_attention_mask"] = encoded.attention_mask
        x["input_ids"] = encoded.ids
        x["type_ids"] = encoded.type_ids

        if LABELKEY in x:
            label = x.pop(LABELKEY)
            if label != config.ignore_id:
                lids = []
                for l in label:
                    lid = label_to_id[l]
                    lids.append(lid)
                x[LABELKEY] = lids

        # now we do other text processors
        if text_processors is not None:
            for proc in text_processors:
                if proc == "matched_sentence_modeling":
                    continue
                proc_func = _data_procecessors.get(proc)
                proc_func(x, **proc_args)

        # now we do label proccesor
        if config.label_processor is not None:
            proc_func = _data_procecessors.get(config.label_processor)
            proc_func(x, **proc_args)

        for k, v in x.items():
            if isinstance(v, list) and not isinstance(v[0], str):
                TORCHCOLS.add(k)
        if LABELKEY in x:
            TORCHCOLS.add(LABELKEY)

        return x

    def __len__(self):
        if self.config.img_first:
            return len(self.uniq_imgs)
        else:
            return len(self.datasets)

    @property
    def special_tokens(self):
        return [
            "[unk]",
            "[sep]",
            "[pad]",
            "[cls]",
            "[mask]",
        ]

    def _handle_image(self, entry):
        proc_args = {"config": self.config}
        if self.config.rand_feats is not None:
            feat_shape = tuple(self.config.rand_feats)
            filepath = entry[RAWIMAGEKEY]
            entry["filepath"] = filepath
            img = torch.rand(feat_shape)
            entry[RAWIMAGEKEY] = img

        elif self.config.extractor is None:
            filepath = entry[RAWIMAGEKEY]
            entry["filepath"] = filepath
            image_preprocessor_name = self.config.image_preprocessor
            image_preprocessor = _image_preprocessors.get(image_preprocessor_name)
            config_dict = self.config.to_dict()
            img = Image.open(filepath)
            img = torch.Tensor(numpy.array(img))
            try:
                img = img.permute((2, 0, 1))
            except Exception:
                img = img.unsqueeze(-1).repeat(1, 1, 3).permute((2, 0, 1))

            # okay, now we have the image the way that we want it. so what now?
            func_dict = collect_args_to_func(image_preprocessor, config_dict)
            img = image_preprocessor(img, **func_dict)["imgs"].squeeze(0)
            entry[RAWIMAGEKEY] = img
        else:
            for k, v in entry.items():
                if isinstance(v, Iterable):
                    if isinstance(v, numpy.ndarray):
                        entry[k] = torch.from_numpy(v)
                    elif isinstance(v, list):
                        entry[k] = torch.tensor(v)
                elif isinstance(v, int) or isinstance(v, float):
                    entry[k] = torch.tensor(v)
            if "roi_features" in entry:
                proc_args["random_feat_func"] = self.random_feat
                entry["roi_features"] = entry["roi_features"][: self.config.max_objects]

        # now we do other image processors
        if self.config.image_processors is not None:
            for proc in self.config.image_processors:
                proc_func = _data_procecessors.get(proc)
                proc_func(entry, **proc_args)

    def random_feat(self):
        rand_ind = random.randint(0, len(self.uniq_imgs) - 1)
        img_id = self.uniq_imgs[rand_ind]
        ts_name, ts_split = self.img2textset[img_id]
        textset = self.textsetdict[ts_name][ts_split]
        is_name, is_split = zip(*textset.data_info[ts_split].items())
        imageset = self.imagesetdict[is_name[0]][is_split[0][0]]
        img_info = imageset.get(img_id)
        if "roi_features" in img_info:
            feat = random.choice(img_info["roi_features"])
            return feat
        else:
            return None

    def random_sent(self):
        rand_ind = random.randint(0, len(self.datasets) - 1)
        text_info = self.datasets[rand_ind]
        rand_sent = text_info[TEXTKEY]
        return rand_sent

    def _map(self, small_textset):
        proc_args = self.processor_args()
        return small_textset.map(
            lambda x: UniversalDataset.text_map_function(x, proc_args=proc_args)
        )

    def _do_map_img_first(self, i):
        img_id = self.uniq_imgs[i]
        ts_name, ts_split = self.img2textset[img_id]
        textset = self.textsetdict[ts_name][ts_split]
        idxs = textset.img_to_rows_map[img_id]
        small_textset = textset.select(idxs)
        img_text_dict = self._map(small_textset)

        img_text_dict.set_format(
            type="torch", output_all_columns=True, columns=list(TORCHCOLS)
        )
        # so what we have to turn into tensors ar
        img_text_dict = img_text_dict[:]
        img_text_dict[IMAGEKEY] = img_id
        return img_text_dict, textset, ts_split, img_id

    def _do_map_text_first(self, i):
        textset, ind = self.datasets.get_textset_and_ind(i)
        small_textset = textset[ind]
        img_id = small_textset["img_id"]
        proc_args = self.processor_args()
        text_info = self.text_map_function(small_textset, proc_args)
        text_info = dict(
            map(
                lambda x: (
                    x[0],
                    torch.tensor(x[1]) if x[0] in TORCHCOLS else x[1],
                ),
                text_info.items(),
            )
        )

        return text_info, img_id, textset

    @torch.no_grad()
    def __getitem__(self, i):
        if self.config.img_first:
            if (
                self.config.test_run
                and self.cache_batch_exists
                and not self.config.overwrite_cache_batch
            ):
                cache_batch = torch.load(self.cache_batch_path)
                return cache_batch

            img_text_dict, textset, ts_split, img_id = self._do_map_img_first(i)
            is_name, is_split = zip(*textset.data_info[ts_split].items())
            imageset = self.imagesetdict[is_name[0]][is_split[0][0]]
            img_info_dict = imageset.get(img_id)
            if isinstance(img_info_dict, str):
                img_info_dict = {RAWIMAGEKEY: img_info_dict}
            self._handle_image(img_info_dict)
            entry = {**img_text_dict, **img_info_dict, "img_id": img_id}
            # entry = img_info_dict
            if not self.cache_batch_exists or self.config.overwrite_cache_batch:
                torch.save(entry, self.cache_batch_path)

            return entry
        else:
            if (
                self.config.test_run
                and self.cache_batch_exists
                and not self.config.overwrite_cache_batch
            ):
                return torch.load(self.cache_batch_path)

            text_info, img_id, textset = self._do_map_text_first(i)
            ts_name, ts_split = self.img2textset[img_id]
            is_name, is_split = zip(*textset.data_info[ts_split].items())
            imageset = self.imagesetdict[is_name[0]][is_split[0][0]]
            img_info_dict = imageset.get(img_id)
            if isinstance(img_info_dict, str):
                img_info_dict = {RAWIMAGEKEY: img_info_dict}
            self._handle_image(img_info_dict)
            entry = {**text_info, **img_info_dict, "img_id": img_id}
            if not self.cache_batch_exists or self.config.overwrite_cache_batch:
                torch.save(entry, self.cache_batch_path)

            return entry

    @property
    def batch_size(self):
        if len(set(self.splits).intersection(self.config.train_aliases)) == 0:
            return self.config.eval_batch_size
        else:
            return self.config.train_batch_size

    @staticmethod
    # unfinished
    def flatten_text(batch, flatten_keys=None):
        if flatten_keys is None:
            flatten_keys = {"input_ids", "type_ids", "text_attention_mask", "label"}
        for f in flatten_keys:
            flattened = None
            key = batch[f]
            for i in key:
                if flattened is None:
                    key[i] = flattened
                else:
                    flattened = torch.cat((flattened, key[i]), dim=0)
            batch[f] = flattened

    @staticmethod
    def transpose_img2txt(batch, img_keys, device=None, max_size=36):
        if isinstance(device, list):
            device = device[0]
        # first we resize image according to how many examples that we need
        n_sents_per_img = [len(i) for i in batch["input_ids"]]
        for img_key in img_keys:
            assert img_key in batch, f"{img_key} not in {list(batch.keys())}"
            imgs = torch.cat(
                [
                    i.unsqueeze(0).expand(min(n, max_size), *i.shape)
                    for i, n in zip(batch.pop(img_key), n_sents_per_img)
                ],
                dim=0,
            )
            batch[img_key] = imgs
            if device is not None:
                batch[img_key] = batch[img_key].to(device)
        # then we resize everything else
        for k in batch:
            if k not in img_keys:
                if isinstance(batch[k][0], torch.Tensor):
                    # here is a part that we want to reduce

                    # ll = []
                    # for i, (j, n) in enumerate(zip(batch[k], n_sents_per_img)):
                    #     l = []
                    batch[k] = torch.cat(
                        [
                            j[: min(max_size, n)]
                            for i, (j, n) in enumerate(zip(batch[k], n_sents_per_img))
                        ],
                        dim=0,
                    )
                    if device is not None:
                        batch[k] = batch[k].to(device)
                elif isinstance(batch[k][0], str):
                    new_v = []
                    # here is also a part that we want to recude
                    for i, n in zip(batch[k], n_sents_per_img):
                        if n >= max_size:
                            n = min(n, max_size)
                        new_v.extend(i * n)
                    batch[k] = new_v
