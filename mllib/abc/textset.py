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
from mllib import utils
from mllib.utils import import_funcs_from_file
from tqdm import tqdm

LABELPROCPATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "Label.py"
)

LABELPROC = import_funcs_from_file(LABELPROCPATH, pkg="mllib.processing")


def set_metadata(tbl, tbl_meta={}):
    fields = []
    for f in tbl.schema.names:
        fields.append(tbl.schema.field_by_name(f))

    tbl_metadata = tbl.schema.metadata
    for k, v in tbl_meta.items():
        tbl_metadata[k] = json.dumps(v).encode("utf-8")

    schema = pyarrow.schema(fields, metadata=tbl_metadata)
    tbl = pyarrow.Table.from_arrays(list(tbl.itercolumns()), schema=schema)

    return tbl


class Textset(ds.Dataset, metaclass=ABCMeta):
    text_reference = "textset"
    dataset_reference = "imageset"
    img_key = "img_id"
    text_key = "text"
    score_key = "score"
    label_key = "label"

    def __init__(
        self,
        arrow_table,
        answer_frequencies,
        img_to_rows_map,
        info=None,
        imageset_files=None,
        raw_files=None,
        split="",
        **kwargs,
    ):
        super().__init__(arrow_table=arrow_table, info=info, **kwargs)
        self._answer_frequencies = answer_frequencies
        self._img_to_rows_map = img_to_rows_map
        self._imageset_files = imageset_files
        self._raw_files = raw_files
        self._split = split
        if raw_files is not None:
            self._raw_map = {s.split("/")[-1].split(".")[0]: s for s in raw_files}

    @staticmethod
    def custom_finalize(writer, close_stream=True):
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
    def _split_handler(split):
        if "valid" in split:
            split = split.replace("valid", "val")
        if "testdev" in split:
            split = split.replace("testdev", "eval")
        elif "dev" in split:
            split = split.replace("dev", "eval")

        if split == "trainval":
            splits = ["train", "val"]
        elif split == "traineval":
            splits = ["train", "eval"]
        elif split in {"train", "eval", "val"}:
            splits = [split]
        elif split is None:
            return ""
        else:
            raise Exception(f"{split} is not a valid split")
        return splits

    # def _check_frequency(self, row, min_freq):
    #     labels_scores = list(
    #         filter(
    #             lambda x: self.get_freq(x[0]) > min_freq,
    #             [
    #                 (l, s)
    #                 for l, s in zip(
    #                     row.get(Textset.label_key), row.get(Textset.score_key)
    #                 )
    #             ],
    #         )
    #     )
    #     if labels_scores:
    #         row[Textset.label_key] = []
    #         row[Textset.score_key] = []
    #         for ls in labels_scores:
    #             row[Textset.label_key].append(ls[0])
    #             row[Textset.score_key].append(ls[1])
    #         return row
    #     else:
    #         return {k: [] for k in row}

    @staticmethod
    def _alias_check(key, aliases, batch_entry):
        if key not in batch_entry:
            if not aliases:
                batch_entry[key] = batch_entry[key]
            else:
                found = False
                for a in aliases:
                    if a in batch_entry:
                        batch_entry[key] = batch_entry[a]
                        found = True
                assert found, (
                    f"{key} could not be set"
                    f"\n add correct image key in {batch_entry.keys()}"
                    f"\n to the set: {aliases}"
                )

    @classmethod
    def extract(
        cls,
        config=None,
        split=None,
        extensions=None,
        path_or_dir=None,
        supervised=True,
        save_to=None,
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
                label_processor = LABELPROC[label_processor]

        if config is None:
            suffixes = extensions
        else:
            suffixes = config.textfile_extensions
        if not isinstance(suffixes, list):
            suffixes = [suffixes]

        input_split = split
        if split is None:
            assert config is not None
            split = config.split
        if extensions is None:
            extensions = ["json", "jsonl"]

        if path_or_dir is None:
            path_or_dir = config.datadirs
            if isinstance(path_or_dir, str):
                path_or_dir = [path_or_dir]
        else:
            path_or_dir = [path_or_dir]

        splits = cls._split_handler(split)
        print(f"searching for input files for splits: {splits}")
        split_dict = {}
        orig_save_to = save_to
        for split in splits:
            if supervised:
                label_dict = Counter()
            else:
                label_dict = None

            cur_row = 0
            imgid2rows = defaultdict(list)

            text_files = []
            for datadir in path_or_dir:
                for suffix in suffixes:
                    for path in Path(datadir).glob(
                        f"**/*.{suffix}",
                    ):
                        path = str(path)
                        if cls.name in path:
                            if split in path:
                                text_files.append(path)

            assert text_files, "could not locate text file locations"
            text_files = list(set(text_files))
            # print(f"extracting from: {text_files}")

            # setup arrow writer
            buffer = pyarrow.BufferOutputStream()
            stream = pyarrow.output_stream(buffer)
            writer = ArrowWriter(features=cls.features, stream=stream)

            # load data
            text_data = []
            # data_type = None
            # data_sub_type = None
            print("loading json files")
            for t in tqdm(text_files):
                data = utils.try_load_json(t)
                # if data_type is None:
                #     data_type = type(next(data))
                # else:
                #     assert isinstance(next(data), data_type)
                # if data_sub_type is None:
                #     data_sub_type = type(next(data))
                # else:
                #     assert isinstance(next(data), data_sub_type)
                text_data.extend(data)

            # custom forward from user
            print("begin extraction")
            batch_entries = cls.forward(
                text_data, label_processor=label_processor, **kwargs
            )

            # pre-checks
            print("writing rows to arrow dataset")
            # this is 4 am coder
            for sub_batch_entries in utils.batcher(batch_entries, n=64):
                flat_entry = None
                for b in sub_batch_entries:
                    imgid2rows[b[Textset.img_key]].append(cur_row)
                    cur_row += 1
                    for l in b["label"]:
                        label_dict.update([l])
                    b = {k: [v] for k, v in b.items()}

                    if flat_entry is None:
                        flat_entry = b
                    else:
                        for k in flat_entry:
                            flat_entry[k].extend(b[k])

                batch = cls.features.encode_batch(flat_entry)
                writer.write_batch(batch)

            dset = datasets.Dataset.from_buffer(buffer.getvalue())
            Textset.custom_finalize(writer, cls)

            # misc.
            dset = pickle.loads(pickle.dumps(dset))
            if orig_save_to is None:
                save_to = os.path.join(datadir, cls.name, f"{split}.arrow")
                os.makedirs(os.path.join(datadir, cls.name), exist_ok=True)
            else:
                assert os.path.isdir(save_to)
                os.makedirs(os.path.join(save_to, cls.name), exist_ok=True)
                save_to = os.path.join(save_to, cls.name, f"{split}.arrow")

            # add extra metadata
            extra_meta = {
                "img_to_rows_map": imgid2rows,
                "answer_frequencies": dict(label_dict),
                # "split": "" if input_split is None else split,
            }
            table = set_metadata(dset._data, tbl_meta=extra_meta)

            # define new writer
            writer = ArrowWriter(path=save_to, schema=table.schema, with_metadata=False)

            # save new table
            writer.write_table(table)
            e, b = Textset.custom_finalize(writer, close_stream=True)
            print(f"Success! You wrote {e} entry(s) and {b >> 20} mb")
            print(f"Located: {save_to}")

            # return class
            arrow_dset = cls(
                raw_files=None,
                arrow_table=table,
                img_to_rows_map=imgid2rows,
                info=dset.info,
                answer_frequencies=dict(label_dict),
                split="" if input_split is None else split,
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
    def from_config(cls, config, split=None):
        if split is None:
            split = config.split
        splits = cls._split_handler(split)
        split_dict = {}
        for split in splits:
            path_dict = Textset.locations(
                config, imageset=cls.imageset, split=split, textset=cls.name
            )
            text_path = [p for p in path_dict["text"] if split in p]
            assert len(text_path) == 1, f"not sure which to load: {text_path}"
            path = text_path[0]
            raw_files = path_dict["raw"]
            imageset_files = path_dict["arrow"]
            print("read table")
            mmap = pyarrow.memory_map(path)
            f = pyarrow.ipc.open_stream(mmap)
            pa_table = f.read_all()
            print("read metadata")
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
            print("load json from metdata")
            img_to_rows_map = json.loads(img_to_rows_map)
            answer_frequencies = json.loads(answer_frequencies)
            print("return textset")

            arrow_dset = cls(
                arrow_table=pa_table,
                img_to_rows_map=img_to_rows_map,
                answer_frequencies=answer_frequencies,
                raw_files=raw_files,
                imageset_files=imageset_files,
                split="" if split is None else split,
            )
            split_dict[split] = arrow_dset

        return split_dict

    @classmethod
    def from_file(cls, path, split=None, raw_files=None, imageset_files=None):
        print("read table")
        mmap = pyarrow.memory_map(path)
        f = pyarrow.ipc.open_stream(mmap)
        pa_table = f.read_all()
        print("read metadata")
        assert "img_to_rows_map".encode("utf-8") in pa_table.schema.metadata.keys()
        assert "answer_frequencies".encode("utf-8") in pa_table.schema.metadata.keys()
        img_to_rows_map = pa_table.schema.metadata["img_to_rows_map".encode("utf-8")]
        answer_frequencies = pa_table.schema.metadata[
            "answer_frequencies".encode("utf-8")
        ]
        print("load json from metdata")
        img_to_rows_map = json.loads(img_to_rows_map)
        answer_frequencies = json.loads(answer_frequencies)
        print("return textset")
        splits = cls._split_handler(split)
        split_dict = {}
        for split in splits:
            arrow_dset = cls(
                arrow_table=pa_table,
                img_to_rows_map=img_to_rows_map,
                answer_frequencies=answer_frequencies,
                raw_files=raw_files,
                imageset_files=imageset_files,
                split="" if split is None else split,
            )
            split_dict[split] = arrow_dset

        return split_dict

    @staticmethod
    def locations(
        config,
        imageset,
        textset,
        split=None,
        basedir=None,
        arrow=None,
        raw=None,
        img_format="jpg",
    ):
        if config is not None:
            img_format = config.img_format
            assert isinstance(img_format, str)
        if arrow is not None:
            assert isinstance(arrow, bool)
        else:
            fields = config.arrow_fields
            if fields is None:
                arrow = True
            elif not fields:
                arrow = False
        if raw is not None:
            assert isinstance(raw, bool)
        else:
            raw = config.use_raw_imgs

        if split is None:
            assert config is not None
            splits = [config.split]
        else:
            splits = Textset._split_handler(split)

        if basedir is None:
            datadirs = config.datadirs
        else:
            datadirs = basedir
        if isinstance(datadirs, list):
            datadirs = [datadirs]

        textset_name = textset
        imageset_name = imageset
        assert textset_name, "Textset must have a name"
        assert imageset_name, "Textset must have a be associated with an imageset"

        raw_files = []
        arrow_files = []
        text_files = []

        if isinstance(datadirs, str):
            datadirs = [datadirs]
        if raw:
            print("search for raw files")
            for datadir in datadirs:
                for path in Path(datadir).glob("**/*.{img_format}"):
                    for split in splits:
                        if (
                            split in str(path).lower()
                            and imageset_name in str(path).lower()
                        ):
                            raw_files.append(str(path))
        if arrow:
            print("search for arrow imageset")
            for datadir in datadirs:
                for path in Path(datadir).glob("**/*.arrow"):
                    for split in splits:
                        if (
                            split in str(path).lower()
                            and imageset_name in str(path).lower()
                        ):
                            arrow_files.append(str(path))

        print("search for arrow textset")
        for datadir in datadirs:
            for path in Path(datadir).glob("**/*.arrow"):
                for split in splits:
                    if split in str(path).lower() and textset_name in str(path).lower():
                        text_files.append(str(path))

        file_dict = {"arrow": arrow_files, "text": text_files, "raw": raw_files}

        return file_dict

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
    def imageset(self) -> str:
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
    def features(self):
        return ds.Features
