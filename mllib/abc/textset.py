import json
import logging
import os
import pickle
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import datasets
import datasets as ds
import pyarrow
from datasets import ArrowWriter
from mllib import IMAGEKEY, LABELKEY, SCOREKEY, TEXTKEY, utils
from mllib.maps import files
from mllib.utils import set_metadata
from tqdm import tqdm

__all__ = ["Textset"]

_labelproc = files.Label()


class Textset(ds.Dataset, metaclass=ABCMeta):
    text_reference = "textset"
    dataset_reference = "imageset"
    img_key = IMAGEKEY
    text_key = TEXTKEY
    score_key = SCOREKEY
    label_key = LABELKEY
    _extensions = ["json", "jsonl"]
    _known_image_formats = ("jpg", "png", "jpeg")
    _base_features = {
        IMAGEKEY: ds.Value("string"),
        TEXTKEY: ds.Value("string"),
        LABELKEY: ds.Sequence(length=-1, feature=ds.Value("string")),
        SCOREKEY: ds.Sequence(length=-1, feature=ds.Value("float32")),
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
        super().__init__(arrow_table=arrow_table, info=info, fingerprint="", **kwargs)
        self._answer_frequencies = answer_frequencies
        self._img_to_rows_map = img_to_rows_map
        self._split = split

    @staticmethod
    def _check_features(default_features):
        feature_dict = default_features
        feature_dict[IMAGEKEY] = Textset._base_features[IMAGEKEY]
        feature_dict[SCOREKEY] = Textset._base_features[SCOREKEY]
        feature_dict[TEXTKEY] = Textset._base_features[TEXTKEY]
        feature_dict[LABELKEY] = Textset._base_features[LABELKEY]
        features = ds.Features(feature_dict)
        return features

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

    def _get_pathes(self, datadirs, split, extractor=None, feat_type="arrow"):
        assert feat_type in ("arrow", "raw")
        if feat_type == "arrow":
            assert extractor is not None
            files = Textset._locate_arrow_files(
                datadirs, self.info, extractor, split=split
            )
        else:
            files = Textset._locate_raw_files(datadirs, self.info, split=split)
        return files

    def get_imgid_to_raw_path(self, datadirs, split):
        files = self._get_pathes(datadirs, split=split, feat_type="raw")
        imgid2path = {Path(p).stem: p for p in files}
        return imgid2path

    def get_arrow_split(self, datadirs, split, extractor):
        files = self._get_pathes(
            datadirs, split=split, extractor=extractor, feat_type="arrow"
        )
        assert (
            len(files) == 1
        ), f"was expecting to find only one file, but found the following: {files}"

        return files[0]

    @staticmethod
    def _locate_arrow_files(datadirs, textset_info, extractor, split=None):
        if split is None:
            split = ""
        image_set_splits = set()
        uniq_datasets = set()
        if split != "":
            for ds_dict in textset_info[split]:
                for dset, splits in ds_dict.items():
                    uniq_datasets.add(dset)
                    for s in splits:
                        image_set_splits.add(s)
        else:
            for split, ds_dict in textset_info.items():
                for dset, splits in ds_dict.items():
                    uniq_datasets.add(dset)
                    for s in splits:
                        image_set_splits.add(s)
        search_files = []
        for dd in datadirs:
            for ud in uniq_datasets:
                for split in image_set_splits:
                    search_files.append(
                        os.path.join(dd, ud, extractor, "{split}.arrow")
                    )
        valid_search_files = list(filter(lambda x: os.path.isfile(x), search_files))
        assert any(valid_search_files), (
            f"attempting to load *.arrow datasets from the following pathes: '{search_files}'"
            f" but none of these are real files"
        )
        print(f"Attempting to load from: {valid_search_files}")
        return valid_search_files

    @staticmethod
    def _locate_raw_files(datadirs, textset_info, split=None):
        raw_files = set()
        if split is None:
            split = ""
        known_suffixes = Textset._known_image_formats
        image_set_splits = set()
        uniq_datasets = set()
        if split != "":
            for ds_dict in textset_info[split]:
                for dset, splits in ds_dict.items():
                    uniq_datasets.add(dset)
                    for s in splits:
                        image_set_splits.add(s)
        else:
            for split, ds_dict in textset_info.items():
                for dset, splits in ds_dict.items():
                    uniq_datasets.add(dset)
                    for s in splits:
                        image_set_splits.add(s)
        search_dirs = []
        for dd in datadirs:
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
            for path in Path(sdir).glob("**/*"):
                if path.sufix in known_suffixes:
                    raw_files.add(str(path))
        return list(raw_files)

    @staticmethod
    def _locate_text_files(path_or_dir, textset_name, split):
        if os.path.isfile(path_or_dir):
            return [path_or_dir]
        text_files = []
        suffixes = Textset._extensions
        for datadir in path_or_dir:
            for suffix in suffixes:
                for path in Path(datadir).glob(
                    f"**/*.{suffix}",
                ):
                    path = str(path)
                    if textset_name in path:
                        if split in path:
                            text_files.append(path)

        assert text_files, "could not locate text file locations"
        text_files = list(set(text_files))
        return text_files

    @classmethod
    def extract(
        cls,
        config=None,
        splits=None,
        path_or_dir=None,
        supervised=True,
        savedir=None,
        min_label_frequency=None,
        label_processor="label_default",
        **kwargs,
    ):
        if supervised:
            if min_label_frequency is None:
                assert config is not None
                min_label_frequency = config.min_label_frequency
            kwargs["min_label_frequency"] = min_label_frequency
            if label_processor is None:
                assert config is not None
                label_processor = config.label_processor
            if not callable(label_processor):
                assert isinstance(label_processor, str), type(label_processor)
                label_processor = _labelproc.get(label_processor)

        if splits is None:
            assert config is not None
            splits = config.train_split
            if isinstance(splits, str):
                splits = [splits]
        else:
            if isinstance(splits, str):
                splits = [splits]

        if path_or_dir is None:
            path_or_dir = config.datadirs
            if isinstance(path_or_dir, str):
                path_or_dir = [path_or_dir]
        else:
            if isinstance(path_or_dir, str):
                path_or_dir = [path_or_dir]

        if savedir is None:
            savedir = os.path.join(path_or_dir[-1], cls.name)
        os.makedirs(savedir, exist_ok=True)

        print(f"searching for input files for splits: {splits}")
        split_dict = {}
        for split in splits:
            label_dict = Counter()
            cur_row = 0
            imgid2rows = defaultdict(list)

            text_files = cls._locate_text_files(
                datadirs=path_or_dir, textset_name=cls.name, slit=split
            )

            features = Textset._check_features(cls.default_features)
            if not supervised:
                features.pop(SCOREKEY)
                features.pop(LABELKEY)
            # setup arrow writer
            buffer = pyarrow.BufferOutputStream()
            stream = pyarrow.output_stream(buffer)
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
                text_data, label_processor=label_processor, **kwargs
            )

            # pre-checks
            print("writing rows to arrow dataset")
            for sub_batch_entries in utils.batcher(batch_entries, n=64):
                flat_entry = None
                for b in sub_batch_entries:
                    if not supervised:
                        b.pop(SCOREKEY, None)
                        b.pop(LABELKEY, None)
                    else:
                        for l in b["label"]:
                            label_dict.update([l])
                    imgid2rows[b[Textset.img_key]].append(cur_row)
                    cur_row += 1
                    b = {k: [v] for k, v in b.items()}

                    if flat_entry is None:
                        flat_entry = b
                    else:
                        for k in flat_entry:
                            flat_entry[k].extend(b[k])

                batch = cls.features.encode_batch(flat_entry)
                writer.write_batch(batch)

            dset = datasets.Dataset.from_buffer(buffer.getvalue())
            Textset._custom_finalize(writer, cls)

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
            e, b = Textset._custom_finalize(writer, close_stream=True)
            print(f"Success! You wrote {e} entry(s) and {b >> 20} mb")
            print(f"Located: {savedir}")

            # return class
            arrow_dset = cls(
                raw_files=None,
                arrow_table=table,
                img_to_rows_map=imgid2rows,
                info=dset.info,
                answer_frequencies=dict(label_dict),
                split=split,
            )
            split_dict[split] = arrow_dset
        return split_dict

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

    @classmethod
    def from_config(cls, config, splits=None):
        if splits is None:
            splits = config.train_split
        if isinstance(splits, str):
            splits = [splits]
        else:
            assert isinstance(splits, list)

        datadirs = config.datadirs
        if isinstance(datadirs, str):
            datadirs = [datadirs]
        else:
            assert isinstance(datadirs, list)

        split_dict = {}
        for split in splits:
            text_path = Textset._locate_text_files(
                datadirs=datadirs, split=split, textset_name=cls.name
            )
            assert len(text_path) == 1, f"not sure which to load: {text_path}"
            text_path = text_path[0]
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
    def from_file(cls, path):
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

    def get_imageset_files(self):
        return self._imageset_files

    def get_from_img(self, img_id, min_freq=14):
        small_dataset = self[self.img_to_rows_map[img_id]]
        img_ids = set(small_dataset.pop(Textset.img_key))
        assert len(img_ids) == 1, img_ids
        img_id = next(iter(img_ids))
        small_dataset[Textset.img_key] = img_id
        return small_dataset

    def get_row(self, i):
        x = self[i]
        x[Textset.text_reference] = self.name
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

    @property
    def split(self):
        return self._split

    @property
    @abstractmethod
    def info(self) -> dict:
        raise Exception("Do not call from abstract class")

    @abstractmethod
    def forward(
        self, text_data: List[dict], label_processor=None, **kwargs
    ) -> List[dict]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        raise Exception("Do not call from abstract class")

    @property
    def labels(self):
        return set(self.answer_frequencies.keys())

    @property
    def num_labels(self):
        return len(self.labels)

    @property
    @abstractmethod
    def default_features(self):
        return dict
