import inspect
import os
import sys
from abc import abstractmethod
from collections import OrderedDict

import datasets as ds
import pyarrow
import torch
import vltk
from datasets import ArrowWriter
from tqdm import tqdm
from vltk.abc.adapter import Adapter
from vltk.configs import ProcessorConfig
from vltk.inspection import collect_args_to_func
from vltk.processing.image import get_rawsize, get_scale, get_size


def clean_imgid_default(imgid):
    return imgid.split("_")[-1].lstrip("0").strip("n")


class VisnExtraction(Adapter):
    _meta_names = [
        "img_to_row_map",
        "model_config",
        "img_to_row_map",
        "dataset",
        "processor_args",
    ]
    _is_feature = True
    _batch_size = 128

    default_processor = None
    model_config = None
    weights = None

    def processor(self, *args, **kwargs):
        return self._processor(*args, **kwargs)

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
                processor_class = ProcessorConfig()
                processor = processor_class.build()
            else:
                processor_class = default_processor
                processor_args = default_processor.to_dict()
            processor = processor_class.build()

        return processor, processor_args

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
        savedir = VisnExtraction._make_save_path(
            searchdir, dataset_name, extractor_name
        )
        processor, processor_args = VisnExtraction._build_image_processor(
            processor_config, processor, cls.default_processor
        )
        schema = VisnExtraction._build_schema(cls.schema, **kwargs)
        model = VisnExtraction._init_model(
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
        files = set(cls._iter_files(searchdirs))
        total_files = len(files)
        for i, path in tqdm(
            enumerate(files),
            file=sys.stdout,
            total=total_files,
        ):
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

            forward_dict = collect_args_to_func(cls.forward, kwargs=kwargs)
            output_dict = cls.forward(model=model, entry=entry, **forward_dict)
            assert isinstance(
                output_dict, dict
            ), "model outputs should be in dict format"
            output_dict[vltk.imgid] = [img_id]

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
            meta_dict = {
                "img_to_row_map": imgid2row,
                "model_config": model_config,
                "dataset": dataset_name if dataset_name is not None else searchdir,
                "processor_args": processor_args,
            }

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

    @property
    @abstractmethod
    def model(self):
        return None
