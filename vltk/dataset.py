# note if we do not immport a pacakage correctly in this class, no loops or exps will be present
from collections import OrderedDict, defaultdict
import os
from copy import deepcopy
from typing import Dict, List
from collections.abc import Iterable
import numpy
from tokenizers import BertWordPieceTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import LxmertTokenizer

# from vltk.dataset.gqa import load_temp_gqa
from vltk.decorators import get_duration
from vltk.maps import dirs, files
from vltk.processing import Image
from vltk import IMAGEKEY, RAWIMAGEKEY, TEXTKEY, LABELKEY, SCOREKEY
from vltk.utils import collect_args_to_func


VOCABPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "libdata/bert-base-uncased-vocab.txt"))
_textsets = dirs.Textsets()
_imagesets = dirs.Imagesets()
_image_preprocessors = files.Image()

class CollatedSets:
    def __init__(self, *args):
        self.args = args
        self.range2listpos = {}
        start = 0
        for i, a in enumerate(args):
            self.range2listpos[range(start, len(a) + start)] = i
            start += len(a)

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
    batch = OrderedDict()
    keys = deepcopy(list(columns[0].keys()))
    for k in keys:
        if k == RAWIMAGEKEY:
            sizes = map(lambda x: x.get(RAWIMAGEKEY).shape[-2:], columns)
            same_size = 1 if len(set(sizes)) == 1 else 0
            if same_size:
                batch[k] = torch.stack([i.pop(k) for i in columns if i is not None])
            else:
                max_h = max(sizes, key=lambda x: x[0])[0]
                max_w = max(sizes, key=lambda x: x[1])[1]
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
            STACK_IGNORE = (LABELKEY, SCOREKEY, TEXTKEY, "type_ids", "input_ids", "text_attention_mask")
            if isinstance(columns[0].get(k), torch.Tensor) and k not in STACK_IGNORE:
                batch[k] = torch.stack([i.pop(k) for i in columns if i is not None])
            else:
                batch[k] = [i.pop(k) for i in columns if i is not None]
    return batch


class UniversalLoader(DataLoader):
    def __init__(self, names, config, splits=None, textsetdict=None, imagesetdict=None):
        splits = config.split if splits is None else splits
        if isinstance(splits, str):
            splits = [splits]
        if set(splits).intersection((config.eval_aliases.union(config.valid_aliases))):
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
            splits=splits,
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
    def __init__(self, names, config, splits=None, textsetdict=None, imagesetdict=None):
        self.config = config
        self.names = names
        self.tokenizer = BertWordPieceTokenizer(VOCABPATH, lowercase=True)
        self.tokenizer.add_special_tokens(self.special_tokens)
        self.tokenizer.enable_truncation(max_length=config.sent_length)
        self.tokenizer.enable_padding(length=config.sent_length)
        self.ts_splits = set()
        self.is_splits = set()
        self.textsets = []
        self.textsetdict = defaultdict(dict)
        self.imagesetdict = defaultdict(dict)


        if textsetdict is None:
            textsetdict = {}
        if imagesetdict is None:
            imagesetdict = {}
        if isinstance(names, str):
            names = [names]
        for name in names:
            for split in splits:
                if split in textsetdict and  split in textsetdict[textset]:
                    textset = textsetdict[name][split]
                else:
                    textset = _textsets.get(name).from_config(config, splits=split)[split]
                self.ts_splits.add(split)
                self.textsets.append(textset)
                print(f"Added Textset {name}: {split}")
                self.textsetdict[name][split] = textset
                is_name, is_split =  zip(*textset.data_info[split].items())
                # select only first entry because multiple image datasets and multiple
                # split per image dataset mapping to one split of a text dataset is not
                # yet supported
                is_name = is_name[0]
                is_split = is_split[0][0]
                if is_name in imagesetdict and is_split in  imagesetdict[is_name]:
                    imageset = imagesetdict[is_name][is_split]
                elif config.extractor is not None:
                    is_path = textset.get_arrow_split(
                        config.datadirs,
                        is_split,
                        config.extractor
                    )
                    imageset = _imagesets.get(config.extractor).from_file(is_path)
                else:
                    imageset = textset.get_imgid_to_raw_path(config.datadirs, is_split)

                print(f"Added Imageset {is_name}: {is_split}")
                self.imagesetdict[is_name][is_split] = imageset

        self.datasets = CollatedSets(*self.textsets)
        self.img2textset = {}
        self.uniq_imgs = set()
        self.uniq_labels = set()
        self.label_to_id = {}
        self.uniq_labels = set()
        for ts_name, ts_splits in self.textsetdict.items():
            for split_name, ts in self.textsetdict[ts_name].items():
                self.uniq_imgs = self.uniq_imgs.union(ts.uniq_imgs)
                self.uniq_labels = self.uniq_labels.union(ts.labels)
                for img in ts.uniq_imgs:
                    self.img2textset[img] = (ts_name, split_name)

        self.uniq_imgs = list(self.img2textset.keys())
        self.uniq_imgs = list(self.img2textset.keys())
        for i, l in enumerate(self.uniq_labels):
            self.label_to_id[l] = i

    def __len__(self):
        if self.config.img_first:
            return len(self.uniq_imgs)
        else:
            return len(self.datasets)

    def _init_tokenizer_args(self):
        self._tokenizer_args = {
            "return_token_type_ids": self.config.return_token_type_ids,
            "return_attention_mask": self.config.return_attention_mask,
            "add_special_tokens": self.config.add_special_tokens,
            "return_tensors": self.config.return_tensors,
        }

    def tokenizer_args(self):
        return self._tokenizer_args

    @property
    def special_tokens(self):
        return [
            "[unk]",
            "[sep]",
            "[pad]",
            "[cls]",
            "[mask]",
            ]

    def _tokenize(self, entry):
        # for text
        text = entry.pop(TEXTKEY)
        if isinstance(text, str):
            encoded_text = self.tokenizer.encode(text)
            type_ids = encoded_text.type_ids
            attention_mask = encoded_text.attention_mask
            token_ids = encoded_text.ids
        else:
            encoded_text = self.tokenizer.encode_batch(text)
            token_ids = []
            type_ids = []
            attention_mask = []
            for enc_text in encoded_text:
                token_ids.append(enc_text.ids)
                type_ids.append(enc_text.type_ids)
                attention_mask.append(enc_text.attention_mask)
        attention_mask = torch.tensor(attention_mask)
        type_ids = torch.tensor(type_ids)
        token_ids = torch.tensor(token_ids)
        entry["input_ids"] = token_ids
        entry["type_ids"] = type_ids
        entry["text_attention_mask"] = attention_mask
        if SCOREKEY in entry:
            score = entry[SCOREKEY]
            if isinstance(score[0], float):
                entry[SCOREKEY] = torch.tensor([deepcopy(s) for s in score])
            else:
                scores = [torch.tensor(deepcopy(s)) for s in score]
                entry[SCOREKEY] = scores
        if LABELKEY in entry:
            label = entry[LABELKEY]
            if isinstance(label[0], str):
                entry[LABELKEY] = torch.tensor([self.label_to_id[l] for l in label])
            else:
                labels = []
                for example in label:
                    sub_labels = []
                    for l in example:
                        sub_labels.append(self.label_to_id[l])
                    labels.append(sub_labels)
                entry[LABELKEY] = [torch.tensor(l) for l in labels]

    def _handle_image(self, entry):
        if self.config.extractor is None:
            filepath = entry[IMAGEKEY]
            image_preprocessor_name = self.config.image_preprocessor
            image_preprocessor = _image_preprocessors.get(image_preprocessor_name)
            config_dict = self.config.to_dict()
            func_dict = collect_args_to_func(
                image_preprocessor,
                config_dict,
            )
            img, (h, w), scales_hw = image_preprocessor(filepath, **func_dict)
            entry[RAWIMAGEKEY] = img
        else:
            for k, v in entry.items():
                if isinstance(v, Iterable):
                    if isinstance(v, numpy.ndarray):
                        entry[k] = torch.from_numpy(v.copy())
                    elif isinstance(v, list):
                        entry[k] = torch.tensor(deepcopy(v))
                elif isinstance(v, int) or isinstance(v, float):
                        entry[k] = torch.tensor(deepcopy(v))
            if "roi_features" in entry:
                entry["roi_features"] = entry["roi_features"][: self.config.max_objects]


    def __getitem__(self, i):
        if self.config.img_first:
            img_id = self.uniq_imgs[i]
            ts_name, ts_split = self.img2textset[img_id]
            textset = self.textsetdict[ts_name][ts_split]
            is_name, is_split = zip(*textset.data_info[ts_split].items())
            imageset = self.imagesetdict[is_name[0]][is_split[0][0]]
            img_text_dict = textset.get_from_img(img_id)
            self._tokenize(img_text_dict)
            img_info_dict = imageset.get(img_id)
            self._handle_image(img_info_dict)
            if isinstance(img_info_dict, str):
                entry = img_text_dict
                entry[IMAGEKEY] = img_info_dict
            else:
                entry = {**img_text_dict, **img_info_dict}
            return entry
        else:
            text_info = self.datasets[i]
            self._tokenize(text_info)
            img_id = text_info[IMAGEKEY]
            ts_name, ts_split = self.img2textset[img_id]
            textset = self.textsetdict[ts_name][ts_split]
            is_name, is_split =  zip(*textset.data_info[ts_split].items())
            imageset = self.imagesetdict[is_name[0]][is_split[0][0]]
            img_info_dict = imageset.get(img_id)
            self._handle_image(img_info_dict)
            if isinstance(img_info_dict, str):
                entry = text_info
                entry[IMAGEKEY] = img_info_dict
            else:
                entry = {**text_info, **img_info_dict}
            return entry

    @property
    def batch_size(self):
        if len(set(self.ts_splits).intersection(self.config.train_aliases)) == 0:
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
