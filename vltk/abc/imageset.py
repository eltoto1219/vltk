import inspect
import json
import logging as logger
import os
import pickle
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from itertools import chain
from pathlib import Path

import datasets
import datasets as ds
import pyarrow
from datasets import ArrowWriter, Split
from tqdm import tqdm
from vltk import IMAGEKEY, IMAGESETPATH
from vltk.inspect import apply_args_to_func, collect_args_to_func, get_classes
from vltk.processing import image as image_proc
from vltk.processing import label as Label
from vltk.utils import set_metadata
from vltk.processing.image import Pipeline

__all__ = ["Imageset", "Imagesets"]

_imageproc = image_proc.Image()


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
    }

    def __init__(self, arrow_table, img_to_row_map, split=None, info=None, **kwargs):
        super().__init__(
            arrow_table=arrow_table, split=split, info=info, fingerprint="", **kwargs
        )
        self._img_to_row_map = img_to_row_map

    def has_id(self, img_id):
        return img_id in self.img_to_row_map

    def shuffle(self):
        print("WARNING: shuffle disabled for imaegeset")

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

    @classmethod
    def extract(
        cls,
        image_preprocessor,
        model,
        dataset_name,
        features=None,
        searchdirs=None,
        savedir=None,
        splits=None,
        config=None,
        img_format="jpg",
        subset_ids=None,
        **kwargs,
    ):
        assert model is not None, "must provide torch model, not none type"

        # update img format if in kwargs or config
        if "img_format" in kwargs:
            img_format = kwargs.get("img_format")
        elif hasattr(config, "img_format"):
            img_format = config.img_format

        # get split name if split name is available
        if isinstance(splits, str):
            splits = [splits]
        elif splits is not None:
            assert isinstance(splits, list)

        if isinstance(searchdirs, str):
            searchdirs = [searchdirs]
        elif isinstance(searchdirs, list):
            pass
        elif searchdirs is None:
            assert config is not None
            # create searchdirs
            print("collecting search dirs")
            datadirs = config.datadirs
            if isinstance(datadirs, list):
                pass
            else:
                assert isinstance(datadirs, str)
                datadirs = [datadirs]
            searchdirs = [os.path.join(x, dataset_name) for x in datadirs]
            gen_savedir = datadirs[-1]
            # find only valid splits
            temp_searchdirs = []
            for i in range(len(searchdirs)):
                sdir = searchdirs.pop(i)
                if os.path.exists(sdir) and splits is not None:
                    sdirs = [
                        os.path.join(sdir, s) for s in os.listdir(sdir) if s in splits
                    ]
                else:
                    sdirs = [os.path.join(sdir, s) for s in os.listdir(sdir)]
                temp_searchdirs.extend(sdirs)
            searchdirs = temp_searchdirs
        else:
            raise Exception

        # get extractor name
        extractor_name = cls.name
        # make savedir if not provdied
        if savedir is None:
            savedir = os.path.join(gen_savedir, dataset_name, extractor_name)
        os.makedirs(savedir, exist_ok=True)
        print(f"will write to directory: {savedir}")

        # turn options from config into dict
        if config is not None:
            presets = config.to_dict()
            presets = {**presets, **kwargs}
        else:
            presets = kwargs

        # check or init image preprocessor
        if image_preprocessor is None:
            # TODO: Fix
            image_preprocessor = Pipeline()
        if callable(image_preprocessor):
            pass
        elif isinstance(image_preprocessor, str):
            image_preprocessor = _imageproc.get(image_preprocessor)
        else:
            raise ValueError("processor must be a string or function")

        # prelim forward checks
        cls._check_forward(image_preprocessor, model, cls.forward)

        # ensure features are in correct format
        features = cls._check_features(features, cls.default_features, presets)

        # setup tracking dicts
        split2buffer = OrderedDict()
        split2stream = OrderedDict()
        split2writer = OrderedDict()
        split2imgid2row = {}
        split2currow = {}

        print(f"SEARCHING recursively in {searchdirs}")
        files = [list(Path(p).rglob(f"*.{img_format}")) for p in searchdirs]
        files = list(chain(*files))
        total_files = len(files)
        print(f"found {total_files} images")
        batch_size = cls._batch_size
        cur_size = 0
        cur_batch = None
        for i, path in tqdm(enumerate(files), total=total_files, file=sys.stdout):
            split = path.parent.name
            img_id = path.stem
            img_id = Label.clean_imgid_default(img_id)
            imgs_left = abs(i + 1 - total_files)
            if splits is not None and split not in splits:
                continue
            if subset_ids is not None and img_id not in subset_ids:
                continue
            # make sure file is not empty
            if path.stat().st_size < 10:
                continue

            # oragnize by split now
            if split not in split2buffer:
                imgid2row = {}
                cur_row = 0
                cur_size = 0
                buffer = pyarrow.BufferOutputStream()
                split2buffer[split] = buffer
                stream = pyarrow.output_stream(buffer)
                split2stream[split] = stream
                writer = ArrowWriter(features=features, stream=stream)
                split2writer[split] = writer
            else:
                # if new split and cur size is not zero, make sure to clear
                if cur_size != 0 and cur_batch is not None:
                    cur_size = 0
                    batch = features.encode_batch(cur_batch)
                    writer.write_batch(batch)
                imgid2row = split2imgid2row[split]
                cur_row = split2currow[split]
                buffer = split2buffer[split]
                stream = split2stream[split]
                writer = split2writer[split]

            if img_id in imgid2row:
                print(f"skipping {img_id}. Already written to table")
            imgid2row[img_id] = cur_row
            cur_row += 1
            split2currow[split] = cur_row
            split2imgid2row[split] = imgid2row
            filepath = str(path)

            # now do model forward
            presets.pop("image_preprocessor", None)
            output_dict = cls.forward(
                filepath=filepath,
                image_preprocessor=image_preprocessor,
                model=model,
                **presets,
            )
            assert isinstance(
                output_dict, dict
            ), "model outputs should be in dict format"
            output_dict["img_id"] = [img_id]

            if cur_batch is None or cur_size == 0:
                cur_batch = output_dict
                cur_size = 1
            else:
                for k, v in cur_batch.items():
                    cur_batch[k].extend(v)
                    cur_size += 1

            # write features
            if cur_size == batch_size or imgs_left < batch_size:
                cur_size = 0
                batch = features.encode_batch(cur_batch)
                writer.write_batch(batch)
            split2imgid2row[split] = imgid2row

        # define datasets
        dsets = []
        splitdict = {}
        print("saving...")
        for (_, writer), (split, b) in zip(split2writer.items(), split2buffer.items()):
            dset = datasets.Dataset.from_buffer(b.getvalue(), split=Split(split))
            dsets.append(dset)
            imgid2row = split2imgid2row[split]
            try:
                writer.finalize(close_stream=False)
            except Exception:
                pass

            # misc.
            dset = pickle.loads(pickle.dumps(dset))

            savefile = os.path.join(savedir, f"{split}.arrow")

            # add extra metadata
            extra_meta = {"img_to_row_map": imgid2row}
            table = set_metadata(dset._data, tbl_meta=extra_meta)
            # define new writer
            writer = ArrowWriter(
                path=savefile, schema=table.schema, with_metadata=False
            )
            # savedir new table
            writer.write_table(table)
            e, b = Imageset.custom_finalize(writer, close_stream=True)
            print(f"Success! You wrote {e} entry(s) and {b >> 20} mb")
            print(f"Located: {savefile}")

            # return class
            arrow_dset = cls(
                arrow_table=table, img_to_row_map=imgid2row, info=dset.info
            )
            splitdict[split] = arrow_dset

        return splitdict

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
    def from_file(cls, path, split=None, name=None):
        if name is not None:
            setattr(cls, "name", name)
        mmap = pyarrow.memory_map(path)
        f = pyarrow.ipc.open_stream(mmap)
        pa_table = f.read_all()
        assert "img_to_row_map".encode("utf-8") in pa_table.schema.metadata.keys()
        img_to_row_map = pa_table.schema.metadata["img_to_row_map".encode("utf-8")]
        img_to_row_map = json.loads(img_to_row_map)
        arrow_dset = cls(
            arrow_table=pa_table, split=split, img_to_row_map=img_to_row_map
        )
        return arrow_dset

    @staticmethod
    def _check_forward(image_preprocessor, model, forward):
        args = str(inspect.formatargspec(*inspect.getargspec(forward)))
        assert "image_preprocessor" in args, (args, type(args))
        assert "filepath" in args, (args, type(args))
        assert "model" in args, (args, type(args))
        assert callable(image_preprocessor), (
            image_preprocessor,
            callable(image_preprocessor),
        )

    @staticmethod
    def _check_features(features, default_features, presets=None):
        if features is None:
            features = default_features
        if presets is None:
            presets = {}
        # check and/or init features
        if features is not None and callable(features):
            features = apply_args_to_func(features, presets)
        if features is None:
            feature_dict = collect_args_to_func(features, presets, mandatory=True)
            feature_dict = default_features(feature_dict)
            raise Exception
            feature_dict[IMAGEKEY] = Imageset._base_features[IMAGEKEY]
            features = ds.Features(feature_dict)
        elif isinstance(features, dict):
            features[IMAGEKEY] = Imageset._base_features[IMAGEKEY]
            features = ds.Features(features)
        else:
            raise Exception(f"incorrect feature type: {type(features)}")
        return features

    @staticmethod
    @abstractmethod
    def forward(filepath, image_preprocessor, model, **kwargs):
        raise Exception("child forward is not being called")

    @abstractmethod
    def default_features(self, *args, **kwargs):
        return dict

    @property
    @abstractmethod
    def name(self):
        return ""
