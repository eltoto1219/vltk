import json
import logging as logger
import os
import pickle
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List

import datasets as ds
import pyarrow
import vltk
from datasets import ArrowWriter, Dataset
from vltk import Features
from vltk.inspection import collect_args_to_func
from vltk.utils.base import set_metadata


class Adapter(ds.Dataset, metaclass=ABCMeta):

    _extensions = ["json", "jsonl"]
    _batch_size = 32
    _base_schema = {vltk.imgid: Features.Imgid}
    _is_annotation = False
    _is_feature = False

    def __init__(
        self,
        arrow_table,
        meta_dict=None,
        split=None,
        info=None,
        fingerprint_off=False,
        **kwargs,
    ):
        if fingerprint_off:
            fingerprint = ""
        else:
            fingerprint = None

        super().__init__(
            arrow_table=arrow_table,
            split=split,
            info=info,
            fingerprint=fingerprint,
            **kwargs,
        )

        if meta_dict is None:
            meta_dict = {}
        for k, v in meta_dict.items():
            setattr(self, "_" + k, v)

    def has(self, img_id):
        return img_id in self.img_to_row_map

    def get(self, img_id):
        return self[self.img_to_row_map[img_id]]

    def shuffle(self):
        raise NotImplementedError

    @property
    def img_to_row_map(self):
        return self._img_to_row_map

    @property
    def name(self):
        return type(self).__name__.lower()

    @property
    def imgids(self):
        return tuple(self._img_to_row_map.keys())

    @property
    def n_imgs(self):
        return len(self.imgids)

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

    @staticmethod
    def _get_valid_search_pathes(searchdir, name=None, splits=None, annodir=None):
        if splits is None:
            splits = vltk.SPLITALIASES
        elif isinstance(splits, str):
            splits = [splits]
        assert os.path.isdir(searchdir)
        if name is not None:
            searchdir = os.path.join(searchdir, name)
            assert os.path.isdir(searchdir)
        if annodir is not None:
            searchdir = os.path.join(searchdir, annodir)
            if not os.path.isdir(searchdir):
                os.makedirs(searchdir, exist_ok=True)
            return searchdir, None
        final_paths = []
        valid_splits = []
        for splt in splits:
            path = os.path.join(searchdir, splt)
            if not os.path.isdir(path):
                continue
            final_paths.append(path)
            valid_splits.append(splt)

        assert final_paths
        return final_paths, valid_splits

    @staticmethod
    def _make_save_path(searchdir, dataset_name, extractor_name):
        if dataset_name is not None:
            savepath = os.path.join(searchdir, dataset_name, extractor_name)
        else:
            savepath = os.path.join(searchdir, extractor_name)
        print(f"will write to {savepath}")
        os.makedirs(savepath, exist_ok=True)
        return savepath

    @staticmethod
    def _iter_files(searchdirs, valid_splits=None):
        if isinstance(searchdirs, str):
            searchdirs = [searchdirs]
        for s in searchdirs:
            for f in os.listdir(s):
                file = Path(os.path.join(s, f))
                if file.stat().st_size > 0:
                    yield Path(os.path.join(s, f))

    @staticmethod
    def _build_schema(features_func, **kwargs):
        feat_args = collect_args_to_func(features_func, kwargs)
        features = features_func(**feat_args)
        default = Adapter._base_schema
        features = {**default, **features}
        return features

    @staticmethod
    def _save_dataset(buffer, writer, savefile, meta_dict, split=None):
        dset = Dataset.from_buffer(buffer.getvalue(), split=ds.Split(split))
        try:
            writer.finalize(close_stream=False)
        except Exception:
            pass
        # misc.
        dset = pickle.loads(pickle.dumps(dset))
        # add extra metadata
        table = set_metadata(
            dset._data, tbl_meta=meta_dict if meta_dict is not None else {}
        )
        # define new writer
        writer = ArrowWriter(path=savefile, schema=table.schema, with_metadata=False)
        # savedir new table
        writer.write_table(table)
        e, b = Adapter._custom_finalize(writer, close_stream=True)
        print(f"Success! You wrote {e} entry(s) and {b >> 20} mb")
        print(f"Located: {savefile}")
        return (table, dset.info, meta_dict)

    @staticmethod
    def _load_one_arrow(filestem, meta_names):
        if ".arrow" not in filestem:
            path = os.path.join(filestem, ".arrow")
        else:
            path = filestem
        assert os.path.isfile(path), f"{path} does not exist"
        mmap = pyarrow.memory_map(path)
        f = pyarrow.ipc.open_stream(mmap)
        pa_table = f.read_all()
        meta_dict = {}
        for n in meta_names:
            assert (
                n.encode("utf-8") in pa_table.schema.metadata.keys()
            ), f"""
            The key {n} is not in the arrow table's metadata: {pa_table.schema.metadata.keys()}
            """
            data_dump = pa_table.schema.metadata[n.encode("utf-8")]
            try:
                data = json.loads(data_dump)
            except Exception:
                data = data_dump
            meta_dict[n] = data
        return (pa_table, meta_dict, path)

    @staticmethod
    def _load_many_arrows(stem, meta_names):
        split_list = []
        for split in vltk.SPLITALIASES:
            temppath = os.path.join(stem, f"{split}.arrow")
            if not os.path.isfile(temppath):
                continue
            pa_table, meta_dict, path = Adapter._load_one_arrow(temppath, meta_names)
            split_list.append((pa_table, meta_dict, split))
        return split_list

    @classmethod
    def load(cls, path, split=None, dataset_name=None):
        meta_names = cls._meta_names
        if ".arrow" in path:
            (pa_table, meta_dict, path) = Adapter._load_one_arrow(path, meta_names)
            return cls(arrow_table=pa_table, split=split, meta_dict=meta_dict)
        # to return visual features
        if dataset_name is not None:
            path = os.path.join(path, dataset_name)
        path = os.path.join(path, cls.__name__.lower())
        if cls._is_annotation:
            path = os.path.join(
                path, f"{vltk.ANNOTATION_DIR}/{vltk.ANNOTATION_DIR}.arrow"
            )
            (pa_table, meta_dict, path) = Adapter._load_one_arrow(path, meta_names)
            return cls(arrow_table=pa_table, split=split, meta_dict=meta_dict)
        elif split is not None:
            path = os.path.join(path, f"{split}.arrow")
            # if cls._is_feature:
            #     path = os.path.join(path, f"{split}.arrow")
            # else:
            #     path = os.path.join(path, split)
            (pa_table, meta_dict, path) = Adapter._load_one_arrow(path, meta_names)
            return cls(arrow_table=pa_table, split=split, meta_dict=meta_dict)
        else:
            arrow_dict = {}
            split_list = Adapter._load_many_arrows(path, meta_names)
            for sl in split_list:
                (pa_table, meta_dict, split) = sl
                arrow_dict[split] = cls(
                    arrow_table=pa_table, split=split, meta_dict=meta_dict
                )
            return arrow_dict

    @abstractmethod
    def forward(*args, **kwargs):
        raise Exception("child forward method is not being called")

    @abstractmethod
    def schema(*args, **kwargs):
        return dict

    @property
    @abstractmethod
    def _meta_names():
        return List
