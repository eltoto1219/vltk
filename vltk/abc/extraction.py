import inspect
import json
import logging as logger
import os
import pickle
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
# from inspect import formatargspec, getargspec
from pathlib import Path

import datasets
import datasets as ds
import pyarrow
import torch
import vltk
from datasets import ArrowWriter, Split
from tqdm import tqdm
from vltk import Features
from vltk.configs import ProcessorConfig
from vltk.inspection import collect_args_to_func, get_classes
from vltk.processing.image import get_rawsize, get_scale, get_size
from vltk.utils import set_metadata

__all__ = ["VizExtractionAdapter", "VizExtractionAdapters"]


def clean_imgid_default(imgid):
    return imgid.split("_")[-1].lstrip("0").strip("n")


class VizExtractionAdapter(ds.Dataset, ABC):
    _batch_size = 10
    _base_schema = {vltk.imgid: Features.imgid}

    default_processor = None
    model_config = None
    weights = None

    def __init__(
        self,
        arrow_table,
        img_to_row_map,
        dataset=None,
        processor_args=None,
        model_config=None,
        split=None,
        info=None,
        **kwargs,
    ):
        super().__init__(
            arrow_table=arrow_table, split=split, info=info, fingerprint="", **kwargs
        )
        self._img_to_row_map = img_to_row_map
        self._dataset = dataset
        self._processor_args = processor_args
        self._config = model_config

    def has(self, img_id):
        return img_id in self.img_to_row_map

    def get(self, img_id):
        return self[self.img_to_row_map[img_id]]

    def shuffle(self):
        raise NotImplementedError

    def processor(self, *args, **kwargs):
        return self._processor(*args, **kwargs)

    def set_img_to_row_map(self, imap):
        self._img_to_row_map = imap

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

    @property
    def processor_args(self):
        return self._processor_args

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

    @property
    def dataset(self):
        return self._dataset

    @property
    def config(self):
        return self._config

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
    def _check_forward(image_preprocessor, forward):
        pass
        args = str(inspect.formatargspec(*inspect.getargspec(forward)))
        assert "entry" in args, (args, type(args))
        assert "model" in args, (args, type(args))
        assert callable(image_preprocessor), (
            image_preprocessor,
            callable(image_preprocessor),
        )

    @staticmethod
    def _get_valid_search_pathes(searchdir, name=None, splits=None):
        if splits is None:
            splits = vltk.SPLITALIASES
        elif isinstance(splits, str):
            splits = [splits]
        assert os.path.isdir(searchdir)
        if name is not None:
            searchdir = os.path.join(searchdir, name)
            assert os.path.isdir(searchdir)
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
    def _iter_files(searchdirs, valid_splits, img_format=None):
        for s in searchdirs:
            for f in os.listdir(s):
                file = Path(os.path.join(s, f))
                if img_format is not None and img_format in f:
                    if file.stat().st_size > 0:
                        yield file
                else:
                    if file.stat().st_size > 0:
                        yield Path(os.path.join(s, f))

    @staticmethod
    def _build_image_processor(config, processor_class, default_processor):
        if config is None:
            processor_args = {}
        else:
            if isinstance(config, dict):
                processor_args = config
            else:
                processor_args = config.to_dict()
        if processor_class is not None:
            processor = processor_class(**processor_args)
        elif config is not None:
            processor = config.build()
        elif config is None:
            if default_processor is None:
                processor_class = ProcessorConfig()
                processor = processor_class.build()
            else:
                processor_class = default_processor
                processor_args = default_processor.to_dict()
            processor = processor_class.build()

        return processor, processor_args

    @staticmethod
    def _build_schema(features_func, **kwargs):
        feat_args = collect_args_to_func(features_func, kwargs)
        features = features_func(**feat_args)
        default = VizExtractionAdapter._base_schema
        features = {**default, **features}
        return features

    @staticmethod
    def _init_model(model_class, model_config, default_config, weights):
        if model_config is None and default_config is not None:
            model_config = default_config

        if model_config is None:
            try:
                model = model_class()
            except Exception:
                raise Exception("Unable to init model without config")
        else:
            try:
                if hasattr(model_class, "from_pretrained") and weights is not None:
                    model = model_class.from_pretrained(weights, model_config)
                else:
                    model = model_class(model_config)
                    if weights is not None:
                        model.load_state_dict(torch.load(weights))
            except Exception:
                raise Exception("Unable to init model with config")
        return model

    @classmethod
    def extract(
        cls,
        searchdir,
        processor_config=None,
        model_config=None,
        splits=None,
        subset_ids=None,
        dataset_name=None,
        img_format="jpg",
        processor=None,
        **kwargs,
    ):

        extractor_name = cls.__name__.lower()
        assert hasattr(cls, "model") and cls.model is not None
        searchdirs, valid_splits = cls._get_valid_search_pathes(
            searchdir, dataset_name, splits
        )
        savedir = VizExtractionAdapter._make_save_path(
            searchdir, dataset_name, extractor_name
        )
        processor, processor_args = VizExtractionAdapter._build_image_processor(
            processor_config, processor, cls.default_processor
        )
        schema = VizExtractionAdapter._build_schema(cls.schema, **kwargs)
        model = VizExtractionAdapter._init_model(
            cls.model, model_config, cls.model_config, cls.weights
        )
        setattr(cls, "model", model)
        # setup tracking dicts
        split2buffer = OrderedDict()
        split2stream = OrderedDict()
        split2writer = OrderedDict()
        split2imgid2row = {}
        split2currow = {}
        # begin search
        print(f"extracting from {searchdirs}")
        batch_size = cls._batch_size
        cur_size = 0
        cur_batch = None
        files = set(cls._iter_files(searchdirs, valid_splits, img_format))
        total_files = len(files)
        # raise Exception(files, total_files)
        for i, path in tqdm(
            enumerate(files),
            file=sys.stdout,
            total=total_files,
        ):
            # path = str(path)
            split = path.parent.name
            img_id = path.stem
            img_id = clean_imgid_default(img_id)
            imgs_left = abs(i + 1 - total_files)
            if split not in valid_splits:
                continue
            if subset_ids is not None and img_id not in subset_ids:
                continue

            # oragnize by split now
            schema = ds.Features(schema)
            if split not in split2buffer:
                imgid2row = {}
                cur_row = 0
                cur_size = 0
                buffer = pyarrow.BufferOutputStream()
                split2buffer[split] = buffer
                stream = pyarrow.output_stream(buffer)
                split2stream[split] = stream
                writer = ArrowWriter(features=schema, stream=stream)
                split2writer[split] = writer
            else:
                # if new split and cur size is not zero, make sure to clear
                if cur_size != 0:
                    cur_size = 0
                    batch = schema.encode_batch(cur_batch)
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

            entry = {vltk.filepath: filepath, vltk.imgid: img_id, vltk.split: split}
            entry[vltk.image] = processor(filepath)
            entry[vltk.size] = get_size(processor)
            entry[vltk.scale] = get_scale(processor)
            entry[vltk.rawsize] = get_rawsize(processor)
            # now do model forward
            output_dict = cls.forward(
                model=model,
                entry=entry,
                **kwargs,
            )
            assert isinstance(
                output_dict, dict
            ), "model outputs should be in dict format"
            output_dict[vltk.imgid] = [img_id]

            if cur_size == 0:
                cur_batch = output_dict
                cur_size = 1
            else:
                # TODO: MASSIVE THIS COULD BE A MASSIVE ERROR
                for k, v in output_dict.items():
                    cur_batch[k].extend(v)
                    cur_size += 1

            # write features
            if cur_size == batch_size or imgs_left < batch_size:
                cur_size = 0
                batch = schema.encode_batch(cur_batch)
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
            extra_meta = {
                "img_to_row_map": imgid2row,
                "model_config": model_config,
                "dataset": dataset_name if dataset_name is not None else searchdir,
                "processor_args": processor_args,
            }
            table = set_metadata(dset._data, tbl_meta=extra_meta)
            # define new writer
            writer = ArrowWriter(
                path=savefile, schema=table.schema, with_metadata=False
            )
            # savedir new table
            writer.write_table(table)
            e, b = VizExtractionAdapter._custom_finalize(writer, close_stream=True)
            print(f"Success! You wrote {e} entry(s) and {b >> 20} mb")
            print(f"Located: {savefile}")

            # return class
            arrow_dset = cls(
                arrow_table=table,
                img_to_row_map=imgid2row,
                info=dset.info,
                dataset=extra_meta["dataset"],
                processor_args=extra_meta["processor_args"],
                model_config=extra_meta["model_config"],
            )
            splitdict[split] = arrow_dset

        return splitdict

    @classmethod
    def load(cls, path, split=None, dataset_name=None):
        # if split is none, load all splits
        if dataset_name is not None:
            path = os.path.join(path, dataset_name)
        name = cls.__name__.lower()
        path = os.path.join(path, name)
        if split is not None:
            path = os.path.join(path, f"{split}.arrow")
            assert os.path.isfile(path), f"{path} does not exist"
            mmap = pyarrow.memory_map(path)
            f = pyarrow.ipc.open_stream(mmap)
            pa_table = f.read_all()
            assert "img_to_row_map".encode("utf-8") in pa_table.schema.metadata.keys()
            img_to_row_map = pa_table.schema.metadata["img_to_row_map".encode("utf-8")]
            dataset = pa_table.schema.metadata["dataset".encode("utf-8")]
            model_config = pa_table.schema.metadata["model_config".encode("utf-8")]
            processor_args = pa_table.schema.metadata["processor_args".encode("utf-8")]
            img_to_row_map = json.loads(img_to_row_map)
            processor_args = json.loads(processor_args)
            arrow_dset = cls(
                arrow_table=pa_table,
                split=split,
                img_to_row_map=img_to_row_map,
                model_config=model_config,
                processor_args=processor_args,
                dataset=dataset,
            )
        else:
            arrow_dset = {}
            for split in vltk.SPLITALIASES:
                temppath = os.path.join(path, f"{split}.arrow")
                if not os.path.isfile(temppath):
                    continue
                mmap = pyarrow.memory_map(temppath)
                f = pyarrow.ipc.open_stream(mmap)
                pa_table = f.read_all()
                assert (
                    "img_to_row_map".encode("utf-8") in pa_table.schema.metadata.keys()
                )
                img_to_row_map = pa_table.schema.metadata[
                    "img_to_row_map".encode("utf-8")
                ]
                dataset = pa_table.schema.metadata["dataset".encode("utf-8")]
                model_config = pa_table.schema.metadata["model_config".encode("utf-8")]
                processor_args = pa_table.schema.metadata[
                    "processor_args".encode("utf-8")
                ]
                img_to_row_map = json.loads(img_to_row_map)
                processor_args = json.loads(processor_args)
                arrow_split = cls(
                    arrow_table=pa_table,
                    split=split,
                    img_to_row_map=img_to_row_map,
                    model_config=model_config,
                    processor_args=processor_args,
                    dataset=dataset,
                )
                arrow_dset[split] = arrow_split
            assert arrow_dset, "no splits found to load arrow dataset"
        return arrow_dset

    @abstractmethod
    def forward(model, entry, **kwargs):
        raise Exception("child forward is not being called")

    @abstractmethod
    def schema(*args, **kwargs):
        return dict

    @property
    @abstractmethod
    def model(self):
        return None


class VizExtractionAdapters:
    def __init__(self):
        if "EXTRACTIONDICT" not in globals():
            global EXTRACTIONDICT
            EXTRACTIONDICT = get_classes(
                vltk.EXTRACTION, ds.Dataset, pkg="vltk.adapters.extraction"
            )

    def avail(self):
        return list(EXTRACTIONDICT.keys())

    def get(self, name):
        return EXTRACTIONDICT[name]

    def add(self, dset):
        EXTRACTIONDICT[dset.__name__.lower()] = dset
