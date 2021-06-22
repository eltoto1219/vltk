import inspect
import os
import resource
import sys
from abc import ABCMeta
from copy import deepcopy

import torch
from datasets import Dataset
# disable logging from datasets
from datasets.utils.logging import set_verbosity_error

# note if we do not immport a pacakage correctly in this class, no loops or exps will be present


__import__("tokenizers")
TOKENIZERS = {
    m[0]: m[1] for m in inspect.getmembers(sys.modules["tokenizers"], inspect.isclass)
}

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (6144, rlimit[1]))

set_verbosity_error()

VOCABPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "libdata/bert-base-uncased-vocab.txt")
).replace("loader/", "")
TOKENIZEDKEY = "encoded"
global TORCHCOLS
TORCHCOLS = set()
os.environ["TOKENIZERS_PARALLELISM"] = "False"


class SplitRangesVision:
    def __init__(self, nested_dict):
        # takes nested dict where the top level is the dataset name
        # the lower level keys are the split names

        self.nested_dict = nested_dict
        self.range2dataset = {}
        self.subrange2dataset = {}
        self.range2split = {}
        uniq_imgs = ()
        start = 0
        dataset_len = 0

        for dataset_name, dataset_dict in nested_dict.items():
            cur_dataset_len = 0
            for split_name, data in dataset_dict.items():
                rng = range(start, len(data) + start)
                self.subrange2dataset[rng] = dataset_name
                self.range2split[rng] = split_name
                start += len(data)
                cur_dataset_len += len(data)

                try:
                    img_ids = data.imgids
                except Exception:
                    img_ids = data.keys()

                img_ids = tuple(img_ids)

                # check once

                uniq_imgs += img_ids

            self.range2dataset[
                range(dataset_len, dataset_len + cur_dataset_len)
            ] = dataset_name
            dataset_len += cur_dataset_len

        self._len = dataset_len
        self.imgs = uniq_imgs

    def __len__(self):
        return self._len

    def __getitem__(self, x):
        if x >= len(self):
            raise IndexError(f"index {x} is out of range 0 to {len(self)}")
        for rng in self.range2split:
            if x in rng:
                visn_split = self.range2split[rng]
                visn_name = self.subrange2dataset[rng]
                return visn_name, visn_split


class SplitRangesVL:
    def __init__(self, nested_dict):
        # takes nested dict where the top level is the dataset name
        # the lower level keys are the split names

        self.nested_dict = nested_dict
        self.range2dataset = {}
        self.subrange2dataset = {}
        self.range2split = {}
        start = 0
        dataset_len = 0
        for dataset_name, dataset_dict in nested_dict.items():
            cur_dataset_len = 0
            for split_name, data in dataset_dict.items():
                rng = range(start, len(data) + start)
                self.subrange2dataset[rng] = dataset_name
                self.range2split[rng] = split_name
                start += len(data)
                cur_dataset_len += len(data)

            self.range2dataset[
                range(dataset_len, dataset_len + cur_dataset_len)
            ] = dataset_name
            dataset_len += cur_dataset_len

        self._len = dataset_len

    def __len__(self):
        return self._len

    def __getitem__(self, x):
        if x >= len(self):
            raise IndexError(f"index {x} is out of range 0 to {len(self)}")
        for rng in self.range2split:
            if x in rng:
                lang_split = self.range2split[rng]
                langset_name = self.subrange2dataset[rng]
                # note does not support matching mulitple vision datasets to a single split
                langset = self.nested_dict[langset_name][lang_split]
                vinset_info = langset.data_info[lang_split]
                visnset_name = next(iter(vinset_info.keys()))
                vinset_splits = tuple(vinset_info.values())[0]
                return (lang_split, langset_name, visnset_name, vinset_splits)


class CollatedVLSets:
    def __init__(self, *args):

        self.args = args
        self.range2listpos = {}
        start = 0
        for i, a in enumerate(args):
            self.range2listpos[range(start, len(a) + start)] = i
            start += len(a)

    # TODO: better solution for more datasets
    def get(self, img_id, return_dataset=False):
        for adapter in self.args:
            try:
                return adapter.get(img_id, return_dataset=return_dataset)
            except KeyError:
                pass
        raise Exception("image id not found in any  visndatasetadapter annotations")

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


class CollatedVisionSets:
    def __init__(self, *args):

        self.args = args
        self.range2listpos = {}
        start = 0
        for i, a in enumerate(args):
            if isinstance(a, ABCMeta):
                continue
            self.range2listpos[range(start, len(a) + start)] = i
            start += len(a)

    # TODO: better solution for more datasets
    def get(self, img_id):
        for adapter in self.args:
            # if not isinstance(adapter, Dataset):
            try:
                return adapter.get(img_id)
            except Exception:
                continue
        return {}

        # raise Exception("image id not found in any  visndatasetadapter annotations")

    def __getitem__(self, x):
        if x >= len(self):
            raise IndexError(f"index {x} is out of range 0 to {len(self)}")
        for rng in self.range2listpos:
            if x in rng:
                listpos = self.range2listpos[rng]
                listind = x - rng.start
                return (
                    self.args[listpos][listind],
                    type(self.args[listpos]).__name__.lower(),
                )

    def __len__(self):
        return sum(map(lambda x: len(x), self.args))

    def __iter__(self):
        return iter(map(lambda x: self[x], range(0, len(self))))


class BaseDataset(Dataset):
    def _init_tokenizer(self, config):
        from_transformers = True
        if isinstance(config.tokenizer, str):
            try:
                self.tokenizer = TOKENIZERS[config.tokenizer](
                    VOCABPATH
                    if config.vocab_file_or_name is None
                    else config.vocab_file_or_name,
                    lowercase=config.lowercase,
                )
                self.special_tokens = set(
                    [
                        self.tokenizer.bos_token,
                        self.tokenizer.eos_token,
                        self.tokenizer.unk_token,
                        self.tokenizer.sep_token,
                        self.tokenizer.pad_token,
                        self.tokenizer.cls_token,
                        self.tokenizer.mask_token,
                    ]
                )
                # self.tokenizer.add_special_tokens(self.special_tokens)
                self.tokenizer.enable_truncation(max_length=config.max_seq_length)
                self.tokenizer.enable_padding(length=config.max_seq_length)
                special_ids = set(
                    [self.tokenizer.token_to_id(t) for t in self.special_tokens]
                )
                self.special_ids = deepcopy(special_ids)
            except KeyError:
                raise Exception(
                    f"{config.tokenizer} not available. Try one of: {TOKENIZERS.keys()}.\
                            OR pass a Tokenizer class from transformers"
                )
            from_transformers = False
        else:
            self.tokenizer = config.tokenizer.from_pretrained(
                VOCABPATH
                if config.vocab_file_or_name is None
                else config.vocab_file_or_name
            )
            self.special_tokens = set(
                [
                    self.tokenizer.bos_token,
                    self.tokenizer.eos_token,
                    self.tokenizer.unk_token,
                    self.tokenizer.sep_token,
                    self.tokenizer.pad_token,
                    self.tokenizer.cls_token,
                    self.tokenizer.mask_token,
                ]
            )
            special_ids = set(
                [self.tokenizer.convert_tokens_to_ids(t) for t in self.special_tokens]
            )
            self.special_ids = deepcopy(special_ids)
        self.from_transformers = from_transformers

        all_ids = tuple(i[1] for i in self.tokenizer.get_vocab().items())
        self.all_ids = all_ids

    def enable_padding(self):
        self.tokenizer.enable_padding(
            length=self.config.lang.max_seq_length,
            direction=self.config.lang.pad_direction,
            pad_id=self.tokenizer.token_to_id(self.tokenizer.pad_token),
        )

    def disable_padding(self):
        self.tokenizer.no_padding()
        pass

    @property
    def batch_size(self):
        if not self.is_train:
            return self.config.eval_batch_size
        else:
            return self.config.train_batch_size

    def try_tensorify(self, entry):
        for k in entry:
            if not isinstance(entry[k], torch.Tensor):
                try:
                    entry[k] = torch.tensor(entry[k])
                except Exception:
                    pass

        return entry

    def _update_placeholders(self, entry):
        raise NotImplementedError
        # if self.all_same_keys:
        #     return
        # entry_keys =
        # placeholders = self.placeholders
        # entry_keys
