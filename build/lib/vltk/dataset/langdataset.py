import inspect
# note if we do not immport a pacakage correctly in this class, no loops or exps will be present
import json
import math
import os
import random
import resource
import sys
from collections import Iterable

import torch
import vltk.vars as vltk
# disable logging from datasets
from datasets.utils.logging import set_verbosity_error
from vltk.dataset.basedataset import BaseDataset, CollatedVLSets
from vltk.processing import LangProcessor, Processors, VisnLangProcessor

__import__("tokenizers")
TOKENIZERS = {
    m[0]: m[1]
    for m in inspect.getmembers(sys.modules["tokenizers"], inspect.isclass)
    if "Tokenizer" in m[0]
}

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (6144, rlimit[1]))

set_verbosity_error()


os.environ["TOKENIZERS_PARALLELISM"] = "False"


class LangDataset(BaseDataset):
    def __init__(
        self,
    ):
        super().__init()
        """
        Implementation for LangDataset is not completed or really developed at all yet.

        This serves as a placeholder class for all relevant pure-text
        processing in the VisnLang Dataset so that I can have that class inherit from
        VisionDataset + LangDataset while I perform this refactoring
        """

    def _init_lang_processors(self, config):
        lang_processors = (
            config.lang_processors if config.lang_processors is not None else []
        )

        lang_processors = [
            x if not isinstance(x, str) else Processors().get(x)
            for x in lang_processors
        ]

        lang_processors = list(
            filter(
                lambda x: x.__bases__[0] == LangProcessor,
                lang_processors,
            )
        )

        self.lang_processors = [
            x(
                tokenizer=self.tokenizer,
                from_transformers=self.from_transformers,
                config=self.config,
                metadata_ids=self.metadata_ids,
            )
            for x in lang_processors
        ]

        self.lang_processor_keys = ()
        for x in self.lang_processor_keys:
            self.lang_processor_keys += tuple(x.keys)

    def run_lang_processors(self, entry, encode_batch=False):
        proc_args = self.lang_processor_args()
        if not encode_batch:
            for processor in self.lang_processors:
                entry = processor(entry, **proc_args)
        else:
            # add ability later to allow this to happen in parallel
            keys, values = zip(*entry.items())
            new_keys = keys
            values_t = list(zip(*values))
            for i, v in enumerate(values_t):
                e = dict(zip(keys, v))
                for processor in self.lang_processors:
                    e = processor(e, **proc_args)
                new_keys, temp_values = zip(*e.items())
                values_t[i] = list(temp_values)
            new_values = list(zip(*values_t))
            entry = dict(zip(new_keys, new_values))
        return entry

    def lang_processor_args(self):
        # max_rand_sents = 1 if not self.config.img_first else 32
        return {
            "tokenizer": self.tokenizer,
            "config": self.config,
            # "random_sents": [self.random_sent() for i in range(max_rand_sents)],
            "special_ids": self.special_ids,
            "metadata_ids": self.metadata_ids,
            "all_ids": self.all_ids,
            "n_ids": len(self.all_ids),
        }

    def tokenize_entry(self, x, encode_batch=False):
        tokenizer = self.tokenizer
        proc_args = self.lang_processor_args()
        from_transformers = self.from_transformers
        if not from_transformers and not encode_batch:
            encoded = tokenizer.encode(x.pop(vltk.text))
            x[vltk.text_attention_mask] = encoded.attention_mask
            x[vltk.input_ids] = encoded.ids
            x[vltk.type_ids] = encoded.type_ids
        elif from_transformers and not encode_batch:
            encoded = tokenizer(
                x.pop(vltk.text),
                padding="max_length",
                truncation="longest_first",
                max_length=proc_args["config"].lang.max_seq_length,
                return_token_type_ids=True,
            )
            x[vltk.text_attention_mask] = encoded["attention_mask"]
            x[vltk.input_ids] = encoded["input_ids"]
            x[vltk.type_ids] = encoded["token_type_ids"]
        elif not from_transformers and encode_batch:
            encoded = tokenizer.encode_batch(x.pop(vltk.text))
            x[vltk.text_attention_mask] = torch.tensor(
                list(map(lambda x: x.attention_mask, encoded))
            )
            x[vltk.input_ids] = torch.tensor(list(map(lambda x: x.ids, encoded)))
            x[vltk.type_ids] = torch.tensor(list(map(lambda x: x.type_ids, encoded)))
        else:
            text = x.pop(vltk.text)
            encoded = tokenizer(
                list(text),
                padding="max_length",
                truncation="longest_first",
                max_length=proc_args["config"].lang.max_seq_length,
                return_token_type_ids=True,
            )
            x[vltk.text_attention_mask] = torch.tensor(encoded["attention_mask"])
            x[vltk.input_ids] = torch.tensor(encoded["input_ids"])
            x[vltk.type_ids] = torch.tensor(encoded["token_type_ids"])
        return x

    def __len__(self):
        return int(math.floor(len(self.datasets) * self.config.percent))

    def random_sent(self):
        rand_ind = random.randint(0, len(self.datasets) - 1)
        text_info = self.datasets[rand_ind]
        rand_sent = text_info[vltk.text]
        return rand_sent

    def _handle_text_label(self, entry, encode_batch=False):
        if vltk.label in entry and not encode_batch:
            label = entry.pop(vltk.label)
            if isinstance(label, torch.Tensor):
                entry[vltk.label] = label
                return entry
            entry[vltk.label] = torch.tensor(self.metadata_ids[vltk.label][label[0]])
            if vltk.score in entry:
                entry[vltk.score] = torch.tensor(entry[vltk.score])
            elif vltk.score in self.max_spanning_cols:
                entry[vltk.score] = torch.ones(entry[vltk.label].shape)
        elif (
            vltk.label not in entry
            and not encode_batch
            and vltk.label in self.max_spanning_cols
        ):
            # this condition is for an entry without label but label is in one dataset
            entry[vltk.label] = torch.tensor(self.config.lang.ignore_id)
            if vltk.score in self.max_spanning_cols:
                entry[vltk.score] = torch.tensor(0).float()

        elif (
            vltk.label not in entry
            and encode_batch
            and vltk.label in self.max_spanning_cols
        ):
            # this condition is for an entry without label but label is in one dataset
            # except across full batch
            entry[vltk.label] = torch.tensor(
                [self.config.lang.ignore_id] * entry[vltk.input_ids].shape[0]
            )
            if vltk.score in self.max_spanning_cols:
                entry[vltk.score] = torch.zeros(entry[vltk.input_ids].shape[0]).float()

        elif (
            vltk.label in entry
            and encode_batch
            and vltk.label in self.max_spanning_cols
        ):
            # TODO dont worry about mulitple labels per thing just yet

            label = entry.pop(vltk.label)
            if isinstance(label, torch.Tensor):
                entry[vltk.label] = label
                return entry
            lids = []
            for l in label:
                lid = self.metadata_ids[vltk.label][l[0]]
                lids.append(lid)
            entry[vltk.label] = torch.tensor(lids)
            if vltk.score in entry:
                entry[vltk.score] = torch.tensor([[v[0] for v in entry[vltk.score]]])
            elif vltk.score in self.max_spanning_cols:
                entry[vltk.score] = torch.ones(entry[vltk.label].shape)
        else:
            pass
        return entry

    def _handle_text_annotations(self, entry, encode_batch=False):
        entry = self.run_lang_processors(entry, encode_batch=encode_batch)

        entry = self.tokenize_entry(entry, encode_batch=encode_batch)
        entry = self._handle_text_label(entry, encode_batch=encode_batch)
        return entry
