import inspect
import os
import sys
from abc import abstractmethod
from collections import OrderedDict

import datasets as ds
import pyarrow
import torch
import vltk.vars as vltk
from datasets import ArrowWriter
from tqdm import tqdm
from vltk.abc.adapter import Adapter
from vltk.configs import VisionConfig
from vltk.inspection import collect_args_to_func
from vltk.processing.image import get_rawsize, get_scale, get_size


class VisnExtraction(Adapter):
    _meta_names = [
        "img_to_row_map",
        "dataset",
        "processor_args",
    ]
    _is_feature = True
    _batch_size = 128

    default_processor = None

    def processor(self, *args, **kwargs):
        return self._processor(*args, **kwargs)

    def align_imgids(self):
        for i in range(len(self)):
            self._img_to_row_map[self[i][vltk.imgid]] = i

    def check_imgid_alignment(self):
        orig_map = self.img_to_row_map
        for i in range(len(self)):
            img_id = self[i][vltk.imgid]
            mapped_ind = orig_map[img_id]
            if mapped_ind != i:
                return False
            self._img_to_row_map[self[i][vltk.imgid]] = i
        return True

    @property
    def processor_args(self):
        return self._processor_args

    @property
    def dataset(self):
        return self._dataset.decode()

    @property
    def config(self):
        return self._config

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
                processor_class = VisionConfig()
                processor = processor_class.build()
            else:
                processor_class = default_processor
                processor_args = default_processor.to_dict()
            processor = processor_class.build()

        return processor, processor_args

    @classmethod
    def extract(
        cls,
        datadir,
        processor_config=None,
        splits=None,
        subset_ids=None,
        dataset=None,
        img_format="jpg",
        processor=None,
        **kwargs,
    ):

        dataset_name = dataset
        searchdir = datadir
        extractor_name = cls.__name__.lower()
        searchdirs, valid_splits = cls._get_valid_search_pathes(
            searchdir, dataset_name, splits
        )
        savedir = VisnExtraction._make_save_path(
            searchdir, dataset_name, extractor_name
        )
        processor, processor_args = VisnExtraction._build_image_processor(
            processor_config, processor, cls.default_processor
        )
        schema = VisnExtraction._build_schema(cls.schema, **kwargs)

        try:
            model, model_config = cls.setup()
        except Exception:
            raise Exception(
                "setup model is supposed to return (`model`, `model_config`) objects, but only returned one object."
            )
        setattr(cls, "model", model)
        # setup tracking dicts
        split2buffer = OrderedDict()
        split2stream = OrderedDict()
        split2writer = OrderedDict()
        split2imgid2row = {}
        split2currow = {}
        split2metadata = {}
        # begin search
        print(f"extracting from {searchdirs}")
        batch_size = cls._batch_size
        cur_size = 0
        cur_batch = None
        files = set(cls._iter_files(searchdirs, iter_imgs=True))
        total_files = len(files)
        for i, path in tqdm(
            enumerate(files),
            file=sys.stdout,
            total=total_files,
        ):
            path_list = path.split("/")
            split = path_list[-2]
            img_id = path_list[-1].split(".")[0]
            imgs_left = abs(i + 1 - total_files)
            if split not in valid_splits:
                continue
            if subset_ids is not None and img_id not in subset_ids:
                continue

            # oragnize by split now
            schema = ds.Features(schema)
            if split not in split2buffer:
                meta_dict = VisnExtraction._init_metadata(schema)
                imgid2row = {}
                cur_row = 0
                cur_size = 0
                buffer = pyarrow.BufferOutputStream()
                split2buffer[split] = buffer
                stream = pyarrow.output_stream(buffer)
                split2stream[split] = stream
                writer = ArrowWriter(features=schema, stream=stream)
                split2writer[split] = writer
                split2metadata[split] = meta_dict
            else:
                # if new split and cur size is not zero, make sure to clear
                if cur_size != 0:
                    cur_size = 0
                    batch = schema.encode_batch(cur_batch)
                    writer.write_batch(batch)
                meta_dict = split2metadata[split]
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
            entry[vltk.img] = processor(filepath)
            entry[vltk.size] = get_size(processor)
            entry[vltk.scale] = get_scale(processor)
            entry[vltk.rawsize] = get_rawsize(processor)
            # now do model forward

            forward_dict = collect_args_to_func(cls.forward, kwargs=kwargs)
            output_dict = cls.forward(model=model, entry=entry, **forward_dict)
            assert isinstance(
                output_dict, dict
            ), "model outputs should be in dict format"
            output_dict[vltk.imgid] = [img_id]
            meta_dict = VisnExtraction._update_metadata(meta_dict, output_dict)

            if cur_size == 0:
                cur_batch = output_dict
                cur_size = 1
            else:
                # TODO: check if this is right
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
        splitdict = {}
        meta_dict = {}
        print("saving...")
        for (_, writer), (split, b) in zip(split2writer.items(), split2buffer.items()):
            savefile = os.path.join(savedir, f"{split}.arrow")
            imgid2row = split2imgid2row[split]
            meta_dict = split2metadata[split]
            meta_dict["img_to_row_map"] = imgid2row
            meta_dict["model_config"] = model_config
            meta_dict["dataset"] = dataset
            meta_dict["processor_args"] = processor_args

            table, info, meta_dict = VisnExtraction._save_dataset(
                b, writer, savefile, meta_dict, split
            )

            # return class
            arrow_dset = cls(
                arrow_table=table,
                split=split,
                info=info,
                meta_dict=meta_dict,
            )
            splitdict[split] = arrow_dset

        return splitdict

    @abstractmethod
    def forward(model, entry, **kwargs):
        raise Exception("child forward is not being called")

    @abstractmethod
    def schema(*args, **kwargs):
        return dict

    @abstractmethod
    def setup():
        return None
