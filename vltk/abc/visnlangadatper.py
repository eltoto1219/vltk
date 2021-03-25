import json
import logging
import os
import pickle
import sys
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import List

import datasets
import datasets as ds
import pyarrow
import vltk
from datasets import ArrowWriter
from tqdm import tqdm
from vltk import VISIONLANG, utils
from vltk.inspection import get_classes
from vltk.processing.label import Label, clean_imgid_default
from vltk.utils import set_metadata

__all__ = ["VisnLangDatasetAdapter", "VisnLangDatasetAdapters"]

_labelproc = Label()


class VisnLangDatasetAdapter(ds.Dataset, metaclass=ABCMeta):
    text_reference = "textset"
    dataset_reference = "visndatasetadapter"
    img_key = vltk.imgid
    text_key = vltk.text
    score_key = vltk.score
    label_key = vltk.label
    _extensions = ["json", "jsonl"]
    _known_image_formats = ("jpg", "png", "jpeg")
    _base_features = {
        vltk.imgid: ds.Value("string"),
        vltk.text: ds.Value("string"),
        vltk.label: ds.Sequence(length=-1, feature=ds.Value("string")),
        vltk.score: ds.Sequence(length=-1, feature=ds.Value("float32")),
    }

    def __init__(
        self,
        arrow_table,
        answer_frequencies,
        img_to_rows_map,
        info=None,
        split="",
        **kwargs,
    ):

        super().__init__(
            arrow_table=arrow_table,
            split=split,
            info=info,
            **kwargs,
        )
        self._answer_frequencies = answer_frequencies
        self._img_to_rows_map = img_to_rows_map
        self._split = split

    @staticmethod
    def _check_features(schema):
        feature_dict = schema
        feature_dict[vltk.imgid] = VisnLangDatasetAdapter._base_features[vltk.imgid]
        feature_dict[vltk.score] = VisnLangDatasetAdapter._base_features[vltk.score]
        feature_dict[vltk.text] = VisnLangDatasetAdapter._base_features[vltk.text]
        feature_dict[vltk.label] = VisnLangDatasetAdapter._base_features[vltk.label]
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
    def _custom_finalize(writer, close_stream=True):
        if writer.pa_writer is None:
            if writer._schema is not None:
                writer._build_writer(writer._schema)
            else:
                raise ValueError(
                    "Please pass `features` or at least one example when writing data"
                )
        if close_stream:
            writer.stream.close()
        logging.info(
            "Done writing %s %s in %s bytes %s.",
            writer._num_examples,
            writer.unit,
            writer._num_bytes,
            writer._path if writer._path else "",
        )
        return writer._num_examples, writer._num_bytes

    @staticmethod
    def _locate_text_set(datadirs, textset_name, split):
        search_files = []
        for dd in datadirs:
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

    @staticmethod
    def _locate_arrow_files(datadirs, textset_data_info, extractor, split=None):
        if split is None:
            split = ""
        image_set_splits = set()
        uniq_datasets = set()
        if isinstance(datadirs, str):
            datadirs = [datadirs]
        if split != "":
            for dset in textset_data_info[split]:
                for split in textset_data_info[split][dset]:
                    uniq_datasets.add(dset)
                    image_set_splits.add(split)
        else:
            for split, ds_dict in textset_data_info.items():
                for dset, splits in ds_dict.items():
                    uniq_datasets.add(dset)
                    for s in splits:
                        image_set_splits.add(s)
        search_files = []
        for dd in datadirs:
            for ud in uniq_datasets:
                for split in image_set_splits:
                    search_files.append(
                        os.path.join(dd, ud, extractor, f"{split}.arrow")
                    )
        valid_search_files = list(filter(lambda x: os.path.isfile(x), search_files))
        assert any(valid_search_files), (
            f"attempting to load *.arrow datasets from the following pathes: '{search_files}'"
            f" but none of these are real files"
        )
        return valid_search_files

    @staticmethod
    def _locate_raw_files(datadirs, textset_data_info, split=None):
        raw_files = set()
        if split is None:
            split = ""
        if isinstance(datadirs, str):
            datadirs = [datadirs]
        known_suffixes = VisnLangDatasetAdapter._known_image_formats
        image_set_splits = set()
        uniq_datasets = set()
        if split != "":
            for dset in textset_data_info[split]:
                for split in textset_data_info[split][dset]:
                    uniq_datasets.add(dset)
                    image_set_splits.add(split)

        else:
            for split, ds_dict in textset_data_info.items():
                for dset, splits in ds_dict.items():
                    uniq_datasets.add(dset)
                    for s in splits:
                        image_set_splits.add(s)
        search_dirs = []
        for dd in datadirs:
            # eg
            for ud in uniq_datasets:
                for split in image_set_splits:
                    search_dirs.append(os.path.join(dd, ud, split))
        valid_search_dirs = list(filter(lambda x: os.path.isdir(x), search_dirs))
        assert any(valid_search_dirs), (
            f"looking for raw images in the following dirs '{search_dirs}'"
            f" but none of these are real directories"
        )
        print(f"locating images in the following dirs: {valid_search_dirs}")
        for sdir in valid_search_dirs:
            for path in tqdm(os.listdir(sdir), file=sys.stdout):
                path = Path(os.path.join(sdir, path))
                if path.suffix[1:] in known_suffixes:
                    raw_files.add(str(path))
        return list(raw_files)

    @staticmethod
    def _locate_text_files(searchdir, textset_name, split):
        searchdir = searchdir[0]
        searchdir = os.path.join(searchdir, textset_name)
        assert os.path.exists(searchdir)
        text_files = []
        suffixes = VisnLangDatasetAdapter._extensions
        for datadir in [searchdir]:
            for suffix in suffixes:
                for path in Path(datadir).glob(
                    f"**/*.{suffix}",
                ):
                    path = str(path)
                    print(path, textset_name)
                    if textset_name in path.lower():
                        if split in path:
                            text_files.append(path)

        if not text_files:
            return None
        text_files = list(set(text_files))
        return text_files

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
            if label_preprocessor is None:
                assert config is not None
                label_preprocessor = config.label_preprocessor
            if not callable(label_preprocessor):
                assert isinstance(label_preprocessor, str), type(label_preprocessor)
                label_preprocessor = _labelproc.get(label_preprocessor)

        if splits is None:
            splits = vltk.SPLITALIASES
        else:
            if isinstance(splits, str):
                splits = [splits]

        if searchdir is None:
            searchdir = config.datadirs
            if isinstance(searchdir, str):
                searchdir = [searchdir]
        else:
            if isinstance(searchdir, str):
                searchdir = [searchdir]

        if savedir is None:
            savedir = os.path.join(searchdir[-1], cls.__name__.lower())
        os.makedirs(savedir, exist_ok=True)

        print(f"searching for input files for splits: {splits}")
        split_dict = {}
        for split in splits:
            label_dict = Counter()
            cur_row = 0
            imgid2rows = defaultdict(list)

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
                    stem = Path(t).stem
                    for f in cls.filters:
                        if f not in stem and t not in temp:
                            temp.append(t)

                text_files = temp

            features = VisnLangDatasetAdapter._check_features(cls.schema)
            if not supervised:
                features.pop(vltk.score)
                features.pop(vltk.label)
            # setup arrow writer
            buffer = pyarrow.BufferOutputStream()
            stream = pyarrow.output_stream(buffer)
            if split == "test" or not supervised:
                if test_features is None:
                    test_features = deepcopy(features)
                    test_features.pop(vltk.score, None)
                    test_features.pop(vltk.label, None)
                writer = ArrowWriter(features=test_features, stream=stream)
            else:
                writer = ArrowWriter(features=features, stream=stream)
            # load data
            text_data = []
            print(f"loading json files from: {text_files}")
            for t in tqdm(text_files):
                data = utils.try_load_json(t)
                text_data.extend(data)

            # custom forward from user
            print("begin extraction")
            batch_entries = cls.forward(
                text_data, split, label_preprocessor=label_preprocessor, **kwargs
            )

            # pre-checks
            print("writing rows to arrow dataset")
            for sub_batch_entries in utils.batcher(batch_entries, n=64):
                flat_entry = None
                for b in sub_batch_entries:
                    if not supervised or split == "test":
                        b.pop(vltk.score, None)
                        b.pop(vltk.label, None)
                    else:
                        for l in b["label"]:
                            label_dict.update([l])
                    imgid2rows[b[VisnLangDatasetAdapter.img_key]].append(cur_row)
                    cur_row += 1
                    b = {k: [v] for k, v in b.items()}

                    if flat_entry is None:
                        flat_entry = b
                    else:
                        for k in flat_entry:
                            flat_entry[k].extend(b[k])

                if split == "test" or not supervised:
                    if test_features is None:
                        test_features = deepcopy(features)
                        test_features.pop(vltk.label, None)
                        test_features.pop(vltk.score, None)
                    batch = test_features.encode_batch(flat_entry)
                    writer.write_batch(batch)
                else:
                    batch = features.encode_batch(flat_entry)
                    writer.write_batch(batch)

            dset = datasets.Dataset.from_buffer(buffer.getvalue())
            VisnLangDatasetAdapter._custom_finalize(writer, cls)

            # misc.
            dset = pickle.loads(pickle.dumps(dset))
            savefile = os.path.join(savedir, f"{split}.arrow")

            # add extra metadata
            extra_meta = {
                "img_to_rows_map": imgid2rows,
                "answer_frequencies": dict(label_dict),
                # "split": "" if input_split is None else split,
            }
            table = set_metadata(dset._data, tbl_meta=extra_meta)

            # define new writer
            writer = ArrowWriter(
                path=savefile, schema=table.schema, with_metadata=False
            )

            # save new table
            writer.write_table(table)
            e, b = VisnLangDatasetAdapter._custom_finalize(writer, close_stream=True)
            print(f"Success! You wrote {e} entry(s) and {b >> 20} mb")
            print(f"Located: {savedir}")

            # return class
            arrow_dset = cls(
                arrow_table=table,
                img_to_rows_map=imgid2rows,
                info=dset.info,
                answer_frequencies=dict(label_dict),
                split=split,
            )
            split_dict[split] = arrow_dset
        return split_dict

    @classmethod
    def from_config(cls, config, splits=None):
        if splits is None:
            splits = config.train_split
        if isinstance(splits, str):
            splits = [splits]
        else:
            assert isinstance(splits, list)

        datadirs = config.datadir
        if isinstance(datadirs, str):
            datadirs = [datadirs]
        else:
            assert isinstance(datadirs, list)

        split_dict = {}
        for split in splits:
            text_path = VisnLangDatasetAdapter._locate_text_set(
                datadirs=datadirs, split=split, textset_name=cls.__name__.lower()
            )
            mmap = pyarrow.memory_map(text_path)
            f = pyarrow.ipc.open_stream(mmap)
            pa_table = f.read_all()
            assert "img_to_rows_map".encode("utf-8") in pa_table.schema.metadata.keys()
            assert (
                "answer_frequencies".encode("utf-8") in pa_table.schema.metadata.keys()
            )
            img_to_rows_map = pa_table.schema.metadata[
                "img_to_rows_map".encode("utf-8")
            ]
            answer_frequencies = pa_table.schema.metadata[
                "answer_frequencies".encode("utf-8")
            ]
            img_to_rows_map = json.loads(img_to_rows_map)
            answer_frequencies = json.loads(answer_frequencies)

            arrow_dset = cls(
                arrow_table=pa_table,
                img_to_rows_map=img_to_rows_map,
                answer_frequencies=answer_frequencies,
                split="" if split is None else split,
            )
            split_dict[split] = arrow_dset

        return split_dict

    @classmethod
    def load(cls, path):
        mmap = pyarrow.memory_map(path)
        f = pyarrow.ipc.open_stream(mmap)
        pa_table = f.read_all()
        assert "img_to_rows_map".encode("utf-8") in pa_table.schema.metadata.keys()
        assert "answer_frequencies".encode("utf-8") in pa_table.schema.metadata.keys()
        img_to_rows_map = pa_table.schema.metadata["img_to_rows_map".encode("utf-8")]
        answer_frequencies = pa_table.schema.metadata[
            "answer_frequencies".encode("utf-8")
        ]
        img_to_rows_map = json.loads(img_to_rows_map)
        answer_frequencies = json.loads(answer_frequencies)
        arrow_dset = cls(
            arrow_table=pa_table,
            img_to_rows_map=img_to_rows_map,
            answer_frequencies=answer_frequencies,
            split="",
        )
        return arrow_dset

    def get(self, img_id, min_freq=14):
        small_dataset = self[self.img_to_rows_map[img_id]]
        img_ids = set(small_dataset.pop(VisnLangDatasetAdapter.img_key))
        assert len(img_ids) == 1, img_ids
        img_id = next(iter(img_ids))
        small_dataset[VisnLangDatasetAdapter.img_key] = img_id
        return small_dataset

    def get_row(self, i):
        x = self[i]
        x[VisnLangDatasetAdapter.text_reference] = self.name
        return x

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

    def get_freq(self, label):
        return self.answer_frequencies[label]

    def _get_pathes(self, datadirs, split, extractor=None, feat_type="arrow"):
        assert feat_type in ("arrow", "raw")
        if feat_type == "arrow":
            assert extractor is not None
            files = VisnLangDatasetAdapter._locate_arrow_files(
                datadirs, self.data_info, extractor, split=split
            )
        else:
            files = VisnLangDatasetAdapter._locate_raw_files(
                datadirs, self.data_info, split=split
            )
        return files

    def get_imgid_to_raw_path(self, datadirs, split):
        files = self._get_pathes(datadirs, split=split, feat_type="raw")
        imgid2path = {clean_imgid_default(Path(p).stem): p for p in files}
        return imgid2path

    def get_arrow_split(self, datadirs, split, extractor):
        files = self._get_pathes(
            datadirs, split=split, extractor=extractor, feat_type="arrow"
        )
        assert (
            len(files) == 1
        ), f"was expecting to find only one file, but found the following: {files}"

        return files[0]

    @property
    def split(self):
        return self._split

    @property
    @abstractmethod
    def data_info(self) -> dict:
        raise Exception("Do not call from abstract class")

    @abstractmethod
    def forward(text_data: List[dict], split: str, **kwargs) -> List[dict]:
        pass

    @property
    def name(self) -> str:
        return type(self).__name__.lower()

    @property
    def labels(self):
        return set(self.answer_frequencies.keys())

    @property
    def num_labels(self):
        return len(self.labels)

    @property
    def uniq_imgs(self):
        return set(self._img_to_rows_map.keys())

    @property
    def img_to_rows_map(self):
        return self._img_to_rows_map

    @property
    def answer_frequencies(self):
        return self._answer_frequencies

    @property
    def raw_file_map(self):
        return self._raw_map

    @property
    @abstractmethod
    def schema(self):
        return dict


class VisnLangDatasetAdapters:
    def __init__(self):
        if "TEXTSETDICT" not in globals():
            global TEXTSETDICT
            TEXTSETDICT = get_classes(
                VISIONLANG, ds.Dataset, pkg="vltk.adapters.visionlang"
            )

    @property
    def dict(self):
        return TEXTSETDICT

    def avail(self):
        return list(TEXTSETDICT.keys())

    def get(self, name):
        return TEXTSETDICT[name]

    def add(self, dset):
        TEXTSETDICT[dset.__name__.lower()] = dset
