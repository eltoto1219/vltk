import json
import logging
import pickle
import tempfile
from abc import ABCMeta, abstractmethod
from collections import Counter
from pathlib import Path

import datasets
import datasets as ds
import pyarrow
from colletions import defaultdict
from datasets import ArrowWriter
from mllib import utils

# LABELPROCPATH = os.path.join(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "features.py"
# )

# LABELPROC = import_funcs_from_file(LABELPROCPATH, pkg="mllib.processing")


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
        writer.pa_writer.close()
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
        dataset_name,
        extensions,
        split,
        path_or_dir,
        supervised=True,
        save_to=None,
        config=None,
        img_key=None,
        label_key=None,
        text_key=None,
    ):

        # setup
        cur_row = 0
        imgid2rows = defaultdict(list)

        if img_key is not None and label_key is not None and text_key is not None:
            aliases = False
        else:
            assert config is not None
            aliases = True
            imgid_aliases = config.imgid_aliases
            text_aliases = config.text_aliases
            label_aliases = config.label_aliases

        if supervised:
            label_dict = Counter()
        else:
            label_dict = None

        if config is None:
            suffixes = extensions
        else:
            suffixes = config.textfile_extensions
        if not isinstance(suffixes, list):
            suffixes = [suffixes]

        input_split = split
        if split is None:
            split = path_or_dir
        if extensions is None:
            extensions = ["json", "jsonl"]

        text_files = filter(
            lambda x: x.split(".")[0] in suffixes and split in x,
            [str(path) for path in Path(path_or_dir).rglob("*")],
        )
        logging.info(f"extracting from: {text_files}")

        # setup arrow writer
        buffer = pyarrow.BufferOutputStream()
        stream = pyarrow.output_stream(buffer)
        writer = ArrowWriter(features=Textset.features, stream=stream)

        # load data
        text_data = []
        data_type = None
        data_sub_type = None
        for t in text_files:
            data = utils.try_load_json(t)
            if data_type is None:
                data_type = type(next(data))
            else:
                assert isinstance(next(data), data_type)
            if data_sub_type is None:
                data_sub_type = type(next(data))
            else:
                assert isinstance(next(data), data_sub_type)
            text_data.extend(data)

        for entry in text_data:
            # custom forward from user
            batch_entry = cls.forward(text_data)
            # pre-checks
            assert isinstance(batch_entry, dict), "ouptut of forward must be a dict"
            assert Textset.score_key in batch_entry, (
                f"label scores with key of {Textset.score_key} "
                "must be provided in the batch_entry"
            )
            assert isinstance(batch_entry[Textset.scores_key], list), (
                "label scores must be by of type list, not:"
                f"{type(batch_entry[Textset.scores_key])}"
            )
            # post-checks
            Textset._alias_check(
                batch_entry=batch_entry,
                aliases={} if not aliases else imgid_aliases,
                key=Textset.img_key if img_key is None else img_key,
            )
            Textset._alias_check(
                batch_entry=batch_entry,
                aliases={} if not aliases else text_aliases,
                key=Textset.text_key if text_key is None else text_key,
            )
            Textset._alias_check(
                batch_entry=batch_entry,
                aliases={} if not aliases else label_aliases,
                key=Textset.label_key if label_key is None else label_key,
            )

            # finishing up / writing batch to table
            batch = Textset.features.encode_batch(
                {k: [v] for k, v in batch_entry.items()}
            )
            imgid2rows[batch_entry[Textset.img_key]].append(cur_row)
            cur_row += 1
            writer.write_batch(batch)

        dset = datasets.Dataset.from_buffer(buffer.getvalue())
        Textset.custom_finalize(writer)

        # misc.
        dset = pickle.loads(pickle.dumps(dset))
        if save_to is None:
            tf = tempfile.NamedTemporaryFile()
            save_to = tf.name

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
        return arrow_dset

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
    def from_config(config, split=None):
        path_dict = Textset.locations(config)
        text_path = path_dict["text"]
        assert len(text_path) == 1, text_path
        return Textset.from_file(
            text_path,
            split=split,
            raw_files=path_dict["raw"],
            imageset_files=path_dict["arrow"],
        )

    @classmethod
    def from_file(cls, path, split=None, raw_files=None, imageset_files=None):
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
            raw_files=raw_files,
            imageset_files=imageset_files,
            split="" if split is None else split,
        )
        return arrow_dset

    @staticmethod
    def locations(
        config, split=None, basedir=None, arrow=None, raw=None, img_format="jpg"
    ):
        if config is not None:
            img_format = config.img_format
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
            split = config.split
        if basedir is None:
            datadirs = config.datadirs
        else:
            datadirs = basedir
        if isinstance(datadirs, list):
            datadirs = [datadirs]

        textset_name = Textset.name
        imageset_name = Textset.imageset
        assert textset_name, "Textset must have a name"
        assert imageset_name, "Textset must have a be associated with an imageset"

        raw_files = []
        arrow_files = []
        text_files = []
        if raw:
            for datadir in datadirs:
                for path in filter(
                    lambda x: imageset_name in str(x) and split in str(x),
                    Path(datadir).rglob(f"*.{img_format}"),
                ):
                    raw_files.append(str(path))
        if arrow:
            for datadir in datadirs:
                for path in filter(
                    lambda x: imageset_name in str(x) and split in str(x),
                    Path(datadir).rglob("*.arrow"),
                ):
                    raw_files.append(str(path))

        for datadir in datadirs:
            for path in filter(
                lambda x: textset_name in str(x) and split in str(x),
                Path(datadir).rglob("*.arrow"),
            ):
                text_files.append(str(path))

        file_dict = {"arrow": arrow_files, "text": text_files, "raw": raw_files}

        return file_dict

    @property
    def raw_file_map(self):
        return self._raw_map

    def get_imageset_files(self):
        return self._imageset_files

    def get_data_by_image(self, img_id):
        return self[self.img_to_row_maps[img_id]]

    def data_as_text_first(self):
        for i in range(len(self)):
            x = self[i]
            img_id = x.pop(Textset.img_key)
            num_entries = len(next(iter(x)))
            for i in range(num_entries):
                sub_entry = {k[i]: v[i] for k, v in x.keys()}
                sub_entry["textset"] = Textset.name
                sub_entry["imageset"] = Textset.imageset
                sub_entry["imgid"] = img_id
                yield x

    def get_freq(self, label):
        return self.answer_frequencies[label]

    @property
    def split(self):
        return self._split

    @property
    def imgid_key(self):
        return "img_id"

    @property
    def text_key(self):
        return "text"

    @property
    def label_key(self):
        return "label"

    @property
    def score_key(self):
        return "score"

    @property
    @abstractmethod
    def imageset(self):
        return ""

    @abstractmethod
    def forward(self, text_entry):
        pass

    @property
    @abstractmethod
    def name(self):
        return ""

    @property
    @abstractmethod
    def features(self):
        return ds.Features
