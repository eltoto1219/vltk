import itertools
import json
import logging as logger
import os
import pickle
import shutil
import sys
import tarfile
import zipfile
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Union

import datasets as ds
import pyarrow
import vltk.vars as vltk
import wget
from datasets import ArrowWriter, Dataset
from vltk.features import Features
from vltk.inspection import collect_args_to_func
from vltk.utils.base import (flatten_stringlist, get_arrow_primitive,
                             set_metadata)

IMGFILES = ("jpeg", "jpg", "png")
SUFFIXES = ("pdf", "json", "jsonl", "csv", "tsv")


class Adapter(ds.Dataset, metaclass=ABCMeta):

    filters: Union[List, None] = None
    """
    Adapter Property that stores a list of string that indicate which patterns
    should be avoided when the Adapter searches for relavant files.
    """
    urls: Union[List, str, None] = None
    """Adapter Property that stores a list of string or just a string which link to
    download urls.
    """
    _extensions = SUFFIXES
    _batch_size = 32
    _base_schema = {vltk.imgid: Features.Imgid()}
    _id_keys = {vltk.imgid, vltk.qid, vltk.text}
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
            if isinstance(k, str):
                k_decoded = k
            else:
                k_decoded = k.decode()
            if k_decoded == "img_to_row_map" or k_decoded == "vocab":
                setattr(self, "_" + k_decoded, v)
            else:
                setattr(self, "meta_" + k_decoded, v)
        setattr(self, "_meta_dict", meta_dict)

    @classmethod
    def download(cls, datadir):
        name = cls.__name__.lower()
        assert (
            cls.urls is not None
        ), f"Cannot re/download {name} because the property: `urls`\
                is not defined. Please define this property as a string or a list of strings\
                in the adapter"
        urls = cls.urls
        if isinstance(urls, str):
            urls = [urls]

        def bar_progress(current, total, width=80):
            progress_message = "Downloading: %d%% [%d / %d] bytes" % (
                current / total * 100,
                current,
                total,
            )
            # Don't use print() as it will print in new line every time.
            sys.stdout.write("\r" + progress_message)
            sys.stdout.flush()

        path = Path(os.path.join(datadir, name))
        temppath = Path(os.path.join(datadir, f"temp_{name}"))
        name_list = []
        try:
            if path.exists():
                temppath.mkdir()
                for f in path.glob("**/*"):
                    newpath = temppath.joinpath(f.name)
                    f.rename(newpath)

                if not path.exists():
                    path.mkdir()
            else:
                os.makedirs(str(path), exist_ok=True)
            for link in urls:
                print(f"Downloading {link}")
                filename = wget.download(link, bar=bar_progress, out=str(path))
                print()

                if ".tar.gz" in filename:
                    with tarfile.open(filename) as tar_ref:
                        print(f"Extracting {filename}")
                        res = tar_ref.extractall(path)
                        # name_list += tar_ref.namelist()
                    # except tarfile.ReadError:
                    #     with zipfile.ZipFile(filename, "r") as zip_ref:
                    #         print(f"Extracting {filename}")
                    #         zip_ref.extractall(path)
                    #         name_list += zip_ref.namelist()
                elif ".zip" in filename:
                    # raise Exception(filename)
                    with zipfile.ZipFile(filename, "r") as zip_ref:
                        print(f"Extracting {filename}")
                        zip_ref.extractall(path)
                        name_list += zip_ref.namelist()
                else:
                    raise Exception(f"{filename} is not a zip file or tarball")
                    # name = Path(filename).name
                    # shutil.move(filename, os.path.join(path, name))
                    # os.remove(filename)
            shutil.rmtree(str(temppath), ignore_errors=True)
            shutil.rmtree(os.path.join(str(path), "__MACOSX"), ignore_errors=True)
        except Exception as e:
            # now remove temppath
            if temppath.exists():
                ###
                for f in temppath.glob("**/*"):
                    newpath = path.joinpath(f.name)
                    f.rename(newpath)
                ###
                if temppath.exists():
                    try:
                        os.rmdir(str(temppath))
                    except Exception:
                        raise Exception(
                            f"{temppath} was made in a failed download attempt. Attempting to remove path,\
                                 but, {temppath} is not empty."
                        )
            raise Exception(
                f"Download Failed: {e}. Moving {str(temppath)} back to {str(path)}"
            )

    @property
    def meta_dict(self):
        return self._meta_dict

    def get_metadata_counters(self):
        if not self._meta_dict:
            return {}

        try:
            schema_dict = collect_args_to_func(type(self).schema, kwargs={})
            feature_dict = {**type(self).schema(**schema_dict), **self._base_features}
        except ValueError:
            feature_dict = {**type(self).schema(), **self._base_features}
        counter_keys = tuple(self._init_metadata(feature_dict).keys())
        counters = {}
        for key in counter_keys:
            if not isinstance(next(iter(self._meta_dict)), str):
                key_encoded = key.encode()
            else:
                key_encoded = key
            if key_encoded in self._meta_dict:
                counters[key] = self._meta_dict[key_encoded]
        return counters

    def has(self, img_id):
        return img_id in self.img_to_row_map

    def get(self, img_id, return_dataset=False):
        if not return_dataset:
            return self[self.img_to_row_map[img_id]]
        else:
            idxs = self.img_to_row_map[img_id]
            assert isinstance(idxs, list)
            return self.select(idxs)

    def get_idx(self, img_id):
        return self.img_to_row_map[img_id]

    def shuffle(self):
        raise NotImplementedError

    def imgid_filter(self, imgids, is_visnlang=True):
        remaining = set(self.imgids).intersection(imgids)

        if is_visnlang:
            idx_groups = dict(
                map(lambda imgid: (imgid, self.get_idx(imgid)), remaining)
            )
            new_map = defaultdict(list)
            idx = 0
            idx_set = []
            for imgid, idxs in idx_groups.items():
                if isinstance(idxs, int):
                    idxs = [idxs]
                    # raise Exception(idx_set, idxs, self)
                idx_set.extend(idxs)
                new_map[imgid] = list(map(lambda x: x[0] + idx, enumerate(idxs)))
                idx += len(idxs)
        else:
            idx_set = list((map(lambda idx: self.get_idx(idx), remaining)))
        filtered_self = self.select(idx_set)
        setattr(filtered_self, "img_to_row_map", self.img_to_row_map)
        setattr(filtered_self, "get_metadata_counters", self.get_metadata_counters)
        setattr(filtered_self, "get", self.get)
        if is_visnlang:
            try:
                setattr(filtered_self, "data_info", self.data_info)
            except Exception:
                pass
            setattr(filtered_self, "_img_to_row_map", new_map)
            try:
                setattr(filtered_self, "imgids", remaining)
            except Exception:
                pass

            # setattr(filtered_self, "_img_to_row_map", new_map)
        else:
            setattr(filtered_self, "_img_to_row_map", dict(zip(remaining, idx_set)))
            setattr(filtered_self, "check_imgid_alignment", self.check_imgid_alignment)
            setattr(filtered_self, "imgids", remaining)

        return filtered_self

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
        assert os.path.isdir(
            searchdir
        ), f"The specificed datadir, {searchdir}, does not exist"
        if name is not None:
            searchdir = os.path.join(searchdir, name)
            assert os.path.isdir(searchdir), f"{searchdir} is not a dir"
        if annodir is not None:
            return searchdir, None
        final_paths = []
        valid_splits = []
        for splt in splits:
            path = os.path.join(searchdir, splt)
            if not os.path.isdir(path):
                continue
            final_paths.append(path)
            valid_splits.append(splt)

        assert final_paths, (searchdir, name, splits, annodir)
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
    def _iter_files(searchdirs, valid_splits=None, iter_imgs=False):
        text_files = []
        if isinstance(searchdirs, str):
            searchdirs = [searchdirs]
        for datadir in searchdirs:
            iterfilter = IMGFILES if iter_imgs else SUFFIXES
            for suffix in iterfilter:
                for path in Path(datadir).glob(
                    f"**/*.{suffix}",
                ):
                    path = str(path)
                    if valid_splits is not None:
                        split_in = False
                        for split in valid_splits:
                            if split in path:
                                split_in = True
                        if split_in:
                            text_files.append(path)
                    else:
                        text_files.append(path)
                    # if textset_name in path.lower():
                    #     if split == "test" and "dev" in path:
                    #         continue
                    #     if split is None or split in path:
                    #         text_files.append(path)

        if not text_files:
            return None
        text_files = list(set(text_files))
        return text_files

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
    def _load_one_arrow(filestem, meta_names, config=None, name=None):
        if ".arrow" not in filestem:
            path = os.path.join(filestem, ".arrow")
        else:
            path = filestem
        if not os.path.isfile(path):
            path = path.replace("/annotations/", "/")
        assert os.path.isfile(path), f"{path} does not exist"
        # this is where we do the re-extract
        # if not os.path.isfile(path) or (
        #     config is not None and config.reextract and name is not None
        # ):
        #     datadir = config.datadir
        mmap = pyarrow.memory_map(path)
        f = pyarrow.ipc.open_stream(mmap)
        pa_table = f.read_all()
        meta_dict = {}
        for n in pa_table.schema.metadata.keys():
            if n.decode() == "huggingface":
                continue

            data_dump = pa_table.schema.metadata[n]
            try:
                data = json.loads(data_dump)
            except Exception:
                data = data_dump
            meta_dict[n] = data
        return (pa_table, meta_dict, path)

    @staticmethod
    def _load_many_arrows(stem, meta_names, config=None):
        split_list = []
        for split in vltk.SPLITALIASES:
            temppath = os.path.join(stem, f"{split}.arrow")
            if not os.path.isfile(temppath):
                continue
            pa_table, meta_dict, path = Adapter._load_one_arrow(
                temppath, meta_names, config=config
            )
            split_list.append((pa_table, meta_dict, split))
        return split_list

    @classmethod
    def load(cls, path, split=None, dataset_name=None, config=None):
        meta_names = cls._meta_names
        if ".arrow" in path:
            (pa_table, meta_dict, path) = Adapter._load_one_arrow(
                path, meta_names, config=config
            )
            return cls(arrow_table=pa_table, split=split, meta_dict=meta_dict)
        # to return visual features
        if dataset_name is not None:
            path = os.path.join(path, dataset_name)
        path = os.path.join(path, cls.__name__.lower())
        if cls._is_annotation:

            temppath = os.path.join(path, f"{vltk.ANNOTATION_DIR}.arrow")
            if not os.path.exists(temppath):
                temppath = os.path.join(
                    path, f"{vltk.ANNOTATION_DIR}/{vltk.ANNOTATION_DIR}.arrow"
                )
            path = temppath
            (pa_table, meta_dict, path) = Adapter._load_one_arrow(
                path, meta_names, config=config
            )
            return cls(arrow_table=pa_table, split=split, meta_dict=meta_dict)
        elif split is not None:
            path = os.path.join(path, f"{split}.arrow")
            (pa_table, meta_dict, path) = Adapter._load_one_arrow(
                path, meta_names, config=config
            )
            return cls(arrow_table=pa_table, split=split, meta_dict=meta_dict)
        else:
            arrow_dict = {}
            split_list = Adapter._load_many_arrows(path, meta_names, config=config)
            for sl in split_list:
                (pa_table, meta_dict, split) = sl
                arrow_dict[split] = cls(
                    arrow_table=pa_table, split=split, meta_dict=meta_dict
                )
            return arrow_dict

    @staticmethod
    def _init_metadata(schema):
        metadata_dict = {}
        for k, v in schema.items():
            if k not in Adapter._id_keys and get_arrow_primitive(v) == "string":
                metadata_dict[k] = Counter()
        return metadata_dict

    @staticmethod
    def _update_metadata(meta_dict, batch_dict):
        for k, v in meta_dict.items():
            if k in batch_dict:
                meta_dict[k].update(flatten_stringlist(batch_dict[k]))
        return meta_dict

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
