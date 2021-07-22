import os
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import List

import datasets
import datasets as ds
import pyarrow
import vltk.vars as vltk
from datasets import ArrowWriter
from tqdm import tqdm
from vltk.abc.adapter import Adapter
from vltk.features import Features
from vltk.inspection import collect_args_to_func
from vltk.utils import base as utils
from vltk.utils.base import get_list_primitive


class VisnLangDataset(Adapter):
    _extensions = ["json", "jsonl"]
    _base_features = {
        vltk.imgid: Features.String(),
        vltk.text: Features.String(),
        # vltk.score: ds.Sequence(length=-1, feature=ds.Value("float32")),
    }
    _meta_names = ["answer_frequencies", "img_to_row_map"]
    _batch_size = 1028

    @staticmethod
    def adjust_imgid(img_id, dataset_name=None, split_name=None):
        """
        Sometimes the image IDS provided in Vision-Language datasets do not map to the
        actual image ID in the reffered vision dataset. If that is the case, implementing
        this optional function will adjust the image id in aims to appropriately mactch
        the corresponging dataset
        """
        return img_id

    @staticmethod
    def _check_features(schema):
        feature_dict = schema
        feature_dict[vltk.imgid] = VisnLangDataset._base_features[vltk.imgid]
        feature_dict[vltk.score] = VisnLangDataset._base_features[vltk.score]
        feature_dict[vltk.text] = VisnLangDataset._base_features[vltk.text]
        feature_dict[vltk.label] = VisnLangDataset._base_features[vltk.label]
        features = ds.Features(feature_dict)
        return features

    @staticmethod
    def _label_handler(label):
        single_label_score = [1]
        if isinstance(label, str):
            return [label], single_label_score
        if isinstance(label, dict):
            if len(label) == 0:
                return [""], single_label_score
            elif len(label) == 1:
                label = next(iter(label))
                assert isinstance(label, str), label
                return [label], single_label_score
            else:
                labels = []
                scores = []
                for lab, score in label.items():
                    assert isinstance(lab, str)
                    score = float(score)
                    labels.append(lab)
                    scores.append(score)
                return labels, scores

    @staticmethod
    def _locate_text_files(searchdir, textset_name, split):
        if isinstance(searchdir, list):
            searchdir = searchdir[0]
        searchdir = os.path.join(searchdir, textset_name)
        assert os.path.exists(searchdir)
        text_files = []
        suffixes = VisnLangDataset._extensions
        for datadir in [searchdir]:
            for suffix in suffixes:
                for path in Path(datadir).glob(
                    f"**/*.{suffix}",
                ):
                    path = str(path)
                    if textset_name in path.lower():
                        if split == "test" and "dev" in path:
                            continue
                        if split is None or split in path:
                            text_files.append(path)

        if not text_files:
            return None
        text_files = list(set(text_files))
        return text_files

    @staticmethod
    def _locate_text_set(datadir, textset_name, split):
        search_files = []
        for dd in [datadir]:
            search_files.append(os.path.join(dd, textset_name, f"{split}.arrow"))
        valid_search_files = list(filter(lambda x: os.path.isfile(x), search_files))
        assert any(valid_search_files), (
            f"attempting to load *.arrow datasets from the following pathes: '{search_files}'"
            f" but none of these are real files"
        )
        assert (
            len(valid_search_files) == 1
        ), "not sure which file to load: {valid_search_files}"
        valid_search_file = valid_search_files[0]
        return valid_search_file

    @classmethod
    def extract(
        cls,
        searchdir,
        config=None,
        splits=None,
        supervised=True,
        savedir=None,
        min_label_frequency=9,
        label_preprocessor="label_default",
        **kwargs,
    ):
        test_features = None
        if supervised:
            if min_label_frequency is None:
                assert config is not None
                min_label_frequency = config.min_label_frequency
            kwargs["min_label_frequency"] = min_label_frequency

        if splits is None:
            splits = vltk.SPLITALIASES
        else:
            if isinstance(splits, str):
                splits = [splits]

        if searchdir is None:
            searchdir = config.datadirs

        if savedir is None:
            savedir = os.path.join(searchdir, cls.__name__.lower())
        os.makedirs(savedir, exist_ok=True)

        print(f"searching for input files for splits: {splits}")
        file_split_dict = {}
        split_file_numbers = {}
        found_any_files = False
        for split in splits:

            text_files = cls._locate_text_files(
                searchdir=searchdir, textset_name=cls.__name__.lower(), split=split
            )
            if text_files is None:
                continue
            if hasattr(cls, "filters"):
                assert isinstance(
                    cls.filters, list
                ), f"filters must be in a list, not type {type(cls.filters)}"
                temp = []
                for i, t in enumerate(text_files):
                    passe_all = True
                    for f in cls.filters:
                        if f in t:
                            passe_all = False
                    if passe_all:
                        temp.append(t)

                text_files = temp
            if text_files:
                found_any_files = True
            file_split_dict[split] = text_files
            split_file_numbers[split] = len(text_files)

        if not found_any_files:
            print(
                "No files pattern matched with corresponding split, falling back to searching all json in top level directory"
            )
            text_files = cls._locate_text_files(
                searchdir=searchdir, textset_name=cls.__name__.lower(), split=None
            )
            if hasattr(cls, "filters"):
                assert isinstance(
                    cls.filters, list
                ), f"filters must be in a list, not type {type(cls.filters)}"
                temp = []
                for i, t in enumerate(text_files):
                    passe_all = True
                    for f in cls.filters:
                        if f in t:
                            passe_all = False
                    if passe_all:
                        temp.append(t)

                text_files = temp
            if not text_files:
                raise Exception("...could not locate any json files")
            file_split_dict["train"] = text_files

        split_dict = {}
        for split, text_files in file_split_dict.items():
            cur_row = 0
            imgid2rows = defaultdict(list)

            if not text_files:
                continue

            schema_dict = collect_args_to_func(cls.schema, kwargs=kwargs)
            features = ds.Features({**cls.schema(**schema_dict), **cls._base_features})
            meta_dict = VisnLangDataset._init_metadata(features)
            if not supervised:
                features.pop(vltk.score)
                features.pop(vltk.label)
            # setup arrow writer
            buffer = pyarrow.BufferOutputStream()
            stream = pyarrow.output_stream(buffer)
            if split == "test" or not supervised:
                if test_features is None:
                    test_features = deepcopy(features)
                    for f in deepcopy(set(test_features.keys())):
                        if f in meta_dict:
                            test_features.pop(f)
                writer = ArrowWriter(features=test_features, stream=stream)
            else:
                writer = ArrowWriter(features=features, stream=stream)
            # load data
            text_data = {}
            print(f"loading json files from: {text_files}")
            for t in tqdm(text_files):
                data = utils.try_load(t)

                text_data[t] = data

            # custom forward from user
            print("begin extraction")

            kwargs["datadir"] = searchdir
            forward_dict = collect_args_to_func(cls.forward, kwargs=kwargs)
            batch_entries = cls.forward(text_data, split, **forward_dict)

            # pre-checks
            print("writing rows to arrow dataset")
            for sub_batch_entries in utils.batcher(batch_entries, n=64):
                flat_entry = None
                for b in sub_batch_entries:
                    # apply adjust image id functio  here
                    vision_dataset_name_and_split = cls.data_info[split]
                    vdset_name = next(iter(vision_dataset_name_and_split.keys()))
                    vdset_split = next(iter(vision_dataset_name_and_split.values()))
                    b[vltk.imgid] = cls.adjust_imgid(
                        b[vltk.imgid], vdset_name, vdset_split
                    )
                    meta_dict = VisnLangDataset._update_metadata(meta_dict, b)

                    imgid2rows[b[vltk.imgid]].append(cur_row)
                    cur_row += 1
                    b = {k: [v] for k, v in b.items()}

                    if flat_entry is None:
                        flat_entry = b
                    else:
                        for k in flat_entry:
                            flat_entry[k].extend(b[k])

                if split == "test" or not supervised:

                    for f in deepcopy(set(flat_entry.keys())):
                        if f not in test_features:
                            flat_entry.pop(f)

                        if (
                            f in flat_entry
                            and get_list_primitive(flat_entry[f]) is None
                        ):
                            flat_entry.pop(f)
                            if f in test_features:
                                test_features.pop(f)

                    batch = test_features.encode_batch(flat_entry)
                    writer.write_batch(batch)
                else:
                    batch = features.encode_batch(flat_entry)
                    writer.write_batch(batch)

            # misc.
            savefile = os.path.join(savedir, f"{split}.arrow")
            meta_dict["img_to_row_map"] = imgid2rows
            meta_dict["split"] = split

            table, info, meta_dict = Adapter._save_dataset(
                buffer, writer, savefile, meta_dict, split
            )

            # return class
            arrow_dset = cls(
                arrow_table=table,
                split=split,
                info=info,
                meta_dict=meta_dict,
            )
            split_dict[split] = arrow_dset
        return split_dict

    def text_iter(self):
        for i in range(len(self)):
            row = self.get_row(i)
            if row:
                yield row

    def text_first(self):
        text_data = []
        for i in tqdm(range(len(self))):
            row = self.get_row(i)
            text_data.append(row)
        return text_data

    def get_frequency(self, label):
        return self.answer_frequencies[label]

    @property
    def split(self):
        try:
            return self._split
        except AttributeError:
            return None

    @property
    @abstractmethod
    def data_info(self) -> dict:
        raise Exception("Do not call from abstract class")

    @abstractmethod
    def forward(text_data: List[dict], split: str, **kwargs) -> List[dict]:
        pass

    @property
    def labels(self):
        return set(self.answer_frequencies.keys())

    @property
    def n_labels(self):
        return len(self.labels)

    @property
    def uniq_imgs(self):
        return set(self._img_to_row_map.keys())

    @property
    def answer_frequencies(self):
        return self._answer_frequencies
