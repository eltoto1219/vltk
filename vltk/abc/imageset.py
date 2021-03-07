import json
import logging as logger
import os
import pickle
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

import datasets
import datasets as ds
import pyarrow
from datasets import ArrowWriter
from tqdm import tqdm
from vltk import ANNOTATION_DIR, IMAGEKEY, IMAGESETPATH, LABELKEY
from vltk.inspect import get_classes
from vltk.processing.label import clean_imgid_default
from vltk.utils import set_metadata

__all__ = ["Imageset", "Imagesets"]


DEFAULT_ANNOS = {}


class Imagesets:
    def __init__(self):
        if "IMAGESETDICT" not in globals():
            global IMAGESETDICT
            IMAGESETDICT = get_classes(IMAGESETPATH, ds.Dataset, pkg="vltk.imageset")

    def avail(self):
        return list(IMAGESETDICT.keys())

    def get(self, name):
        return IMAGESETDICT[name]

    def add(self, name, dset):
        IMAGESETDICT[name] = dset


class Imageset(ds.Dataset, ABC):
    _batch_size = 10
    _base_features = {
        IMAGEKEY: ds.Value("string"),
        LABELKEY: ds.Sequence(length=-1, feature=ds.Value("string")),
    }

    def __init__(
        self, arrow_table, img_to_row_map, object_frequencies, info=None, **kwargs
    ):
        super().__init__(arrow_table=arrow_table, info=info, fingerprint="", **kwargs)
        self._img_to_row_map = img_to_row_map
        self._object_frequencies = object_frequencies

    def has_id(self, img_id):
        return img_id in self.img_to_row_map

    def shuffle(self):
        print("WARNING: shuffle disabled for imaegeset")

    @staticmethod
    def _custom_finalize(writer, close_stream=True):
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
        logger.info(
            "Done writing %s %s in %s bytes %s.",
            writer._num_examples,
            writer.unit,
            writer._num_bytes,
            writer._path if writer._path else "",
        )
        return writer._num_examples, writer._num_bytes

    @property
    def img_to_row_map(self):
        return self._img_to_row_map

    def get(self, img_id):
        return self[self.img_to_row_map[img_id]]

    @staticmethod
    def get_valid_search_pathes(searchdirs, name, anno_dir):
        if isinstance(searchdirs, str):
            searchdirs = [searchdirs]
        for p in searchdirs:
            if not os.path.isdir(p):
                searchdirs.remove(p)
        assert searchdirs
        new_searchdirs = []
        for p in searchdirs:
            p1 = os.path.join(p, name)
            if not os.path.isdir(p1):
                continue
            p2 = os.path.join(p1, anno_dir)
            if os.path.isdir(p2):
                new_searchdirs.append(p2)

        assert new_searchdirs
        return new_searchdirs

    @staticmethod
    def iter_files(searchdirs, data_format=None):
        for s in searchdirs:
            for f in os.listdir(s):
                if data_format is not None:
                    if data_format in f:
                        file = Path(os.path.join(s, f))
                        if file.stat().st_size > 10:
                            yield file
                else:
                    file = Path(os.path.join(s, f))
                    if file.stat().st_size > 10:
                        yield Path(os.path.join(s, f))

    # multiple things that we can do with this mehtod, we can just extract other random data
    # for each imageset

    @staticmethod
    def files(path):
        files = {}
        for i in os.listdir(path):
            fp = os.path.join(path, i)
            iid = clean_imgid_default(i.split(".")[0])
            files[iid] = fp
        return files

    @classmethod
    def extract(
        cls,
        searchdirs,
        savedir=None,
        data_format="jpg",
        **kwargs,
    ):

        feature_dict = {**cls.default_features(**kwargs), **cls._base_features}
        # lets work on doing the annotations first
        total_annos = {}
        searchdirs = cls.get_valid_search_pathes(
            searchdirs, cls.__name__.lower(), ANNOTATION_DIR
        )
        files = cls.iter_files(searchdirs)
        # get into right format
        json_files = []
        print("loading annotation")
        for anno_file in tqdm(files):
            if "json" not in str(anno_file):
                continue
            if "caption" not in str(anno_file) and "question" not in str(anno_file):
                anno_data = json.load(open(str(anno_file)))
                json_files.append((str(anno_file), anno_data))

        total_annos = cls.forward(json_files)

        # now write
        print("write data")
        writer, buffer, imgid2row, object_dict = cls._write_batches(
            total_annos, feature_dict, cls._batch_size
        )
        print("save data")
        if savedir is None:
            savedir = searchdirs[-1]

        extra_meta = {"img_to_row_map": imgid2row, "object_frequencies": object_dict}
        cls._write_data(writer, buffer, savedir, extra_meta)

    @classmethod
    def load_imgid2path(cls, datadir, split):
        name = cls.__name__.lower()
        path = os.path.join(datadir, name, split)
        return Imageset.files(path)

    @staticmethod
    def _write_batches(annos, feature_dict, batch_size):
        object_dict = Counter()
        features = ds.Features(feature_dict)
        imgid2row = {}
        cur_size = 0
        cur_row = 0
        buffer = pyarrow.BufferOutputStream()
        stream = pyarrow.output_stream(buffer)
        writer = ArrowWriter(features=features, stream=stream)
        n_files = len(annos)
        for i, entry in enumerate(annos):
            imgs_left = abs(i + 1 - n_files)
            img_id = entry[IMAGEKEY]
            object_dict.update(entry[LABELKEY])
            if img_id in imgid2row:
                print(f"skipping {img_id}. Already written to table")
            imgid2row[img_id] = cur_row
            cur_row += 1
            if cur_size == 0:
                for k, v in entry.items():
                    entry[k] = [v]
                cur_batch = entry
                cur_size = 1
            else:

                for k, v in entry.items():
                    cur_batch[k].append(v)
                cur_size += 1

            # write features
            if cur_size == batch_size or imgs_left < batch_size:
                cur_size = 0
                batch = features.encode_batch(cur_batch)
                writer.write_batch(batch)

        return writer, buffer, imgid2row, object_dict

    @property
    def labels(self):
        return set(self._object_frequencies.keys())

    @staticmethod
    def _write_data(writer, buffer, savedir, extra_meta):
        print("saving...")
        dset = datasets.Dataset.from_buffer(buffer.getvalue())
        try:
            writer.finalize(close_stream=False)
        except Exception:
            pass
        # misc.
        dset = pickle.loads(pickle.dumps(dset))
        savefile = os.path.join(savedir, "annotations.arrow")

        # add extra metadata
        table = set_metadata(dset._data, tbl_meta=extra_meta)
        # define new writer
        writer = ArrowWriter(path=savefile, schema=table.schema, with_metadata=False)
        # savedir new table
        writer.write_table(table)
        e, b = Imageset._custom_finalize(writer, close_stream=True)
        print(f"Success! You wrote {e} entry(s) and {b >> 20} mb")
        print(f"Located: {savefile}")

    def set_img_to_row_map(self, imap):
        self._img_to_row_map = imap

    @property
    def get_imgids(self):
        return set(self._img_to_row_map.keys())

    def num_imgs(self):
        return len(self.get_imgids)

    def align_imgids(self):
        for i in range(len(self)):
            self._img_to_row_map[self[i]["img_id"]] = i

    def check_imgid_alignment(self):
        orig_map = self.img_to_row_map
        for i in range(len(self)):
            img_id = self[i]["img_id"]
            mapped_ind = orig_map[img_id]
            if mapped_ind != i:
                return False
            self._img_to_row_map[self[i]["img_id"]] = i
        return True

    @classmethod
    def from_file(cls, *args, **kwargs):
        return Imageset.load(cls, *args, **kwargs)

    @staticmethod
    def _load_handler(path, name, split=None, extractor=None, annotations=False):
        name = name.lower()
        if os.path.isfile(path):
            return path
        path = os.path.join(path, name)
        assert os.path.exists(path), f"No such {path}"

        if extractor is not None and split is not None:
            path = os.path.join(path, f"{split}.arrow")
            assert os.path.exists(path), f"No such file {path}"
            return path
        elif annotations:
            path = os.path.join(path, ANNOTATION_DIR)
            assert os.path.exists(path), f"No such {path}"
            path = os.path.join(path, "annotations.arrow")
            assert os.path.exists(path), f"No such file {path}"
            return path
        elif split is None:
            path = os.path.join(path, split)
            return Imageset.files(path)

    @classmethod
    def load(cls, path, split=None, extractor=None, annotations=None):
        out = Imageset._load_handler(
            path,
            cls.__name__.lower(),
            split=split,
            extractor=extractor,
            annotations=annotations,
        )
        if isinstance(out, dict):
            return out
        mmap = pyarrow.memory_map(out)
        f = pyarrow.ipc.open_stream(mmap)
        pa_table = f.read_all()
        assert "img_to_row_map".encode("utf-8") in pa_table.schema.metadata.keys()
        assert "object_frequencies".encode("utf-8") in pa_table.schema.metadata.keys()
        img_to_row_map = pa_table.schema.metadata["img_to_row_map".encode("utf-8")]
        img_to_row_map = json.loads(img_to_row_map)
        object_frequencies = pa_table.schema.metadata[
            "object_frequencies".encode("utf-8")
        ]
        object_frequencies = json.loads(object_frequencies)
        arrow_dset = cls(
            arrow_table=pa_table,
            img_to_row_map=img_to_row_map,
            object_frequencies=object_frequencies,
        )
        return arrow_dset

    def _name(self):
        return type(self).__name__.lower()

    @staticmethod
    @abstractmethod
    def forward(filepath, image_preprocessor, model, **kwargs):
        raise Exception("child forward is not being called")

    @abstractmethod
    def default_features(self, *args, **kwargs):
        return dict
