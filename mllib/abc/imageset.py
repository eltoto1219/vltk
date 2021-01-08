import inspect
import json
import logging as logger
import os
import pickle
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

import datasets
import datasets as ds
import pyarrow
from datasets import ArrowWriter, Features, Split
from mllib.maps import files
from mllib.utils import get_func_signature, set_metadata
from tqdm import tqdm

__all__ = ["Imageset"]

_featureproc = files.Feature()
_imageproc = files.Image()


class Imageset(ds.Dataset, ABC):
    def __init__(self, arrow_table, img_to_row_map, split=None, info=None, **kwargs):
        super().__init__(arrow_table=arrow_table, split=split, info=info, **kwargs)
        self._img_to_row_map = img_to_row_map

    def has_id(self, img_id):
        return img_id in self.img_to_row_map

    def shuffle(
        self,
        seed,
        generator,
        keep_in_memory,
        load_from_cache_file,
        indices_cache_file_name,
        writer_batch_size,
        new_fingerprint,
    ):
        super().shuffle(
            seed=seed,
            generator=generator,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            indices_cache_file_name=indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=new_fingerprint,
        )
        self.align_imgids()

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

    def get_image(self, img_id):
        return self[self.img_to_row_map[img_id]]

    @classmethod
    def extract(
        cls,
        image_preprocessor,
        model,
        features,
        dataset_name,
        config=None,
        path=None,
        img_format="jpg",
        subset_ids=None,
        save_to=None,
        **kwargs,
    ):

        if config is not None:
            presets = config.to_dict()
            if kwargs is not None:
                for k, v in kwargs.items():
                    presets[k] = v
        else:
            presets = kwargs

        # will save in specified path or in the datadir specified in config
        if save_to is None:
            dd = config.datadirs
            if isinstance(dd, str):
                datadir = dd
            else:
                datadir = config.datadirs[-1]
        else:
            assert os.path.isdir(save_to) or not os.path.exists(save_to)
            datadir = os.path.join(save_to)

        print(f"SEARCHING recursively in {datadir}")

        if callable(image_preprocessor):
            pass
        elif isinstance(image_preprocessor, str):
            image_preprocessor = _imageproc.get(image_preprocessor)
        else:
            raise ValueError("processor must be a string or function")

        assert model is not None, "must specify model"
        if path is not None:
            assert os.path.isdir(path), "dir does not exist, images elsewhere?"
        else:
            path = datadir

        cls._check_forward(image_preprocessor, model, cls.forward)

        if isinstance(features, str):
            feature_func = _featureproc.get(features)
            sig_dict = get_func_signature(feature_func)
            feat_options = {}
            for k in sig_dict.keys():
                assert k in presets, (
                    f"user must specify {k} in config or kwargs"
                    " in the the Imageset.extract method"
                    f"\nThe following are required: {list(sig_dict.keys())}"
                )
                feat_options[k] = presets.pop(k)
            features = feature_func(**feat_options)
        elif not isinstance(features, Features):
            raise Exception(
                "provide string mapping to features, or the features themselves"
            )

        assert "img_id" in features

        split2buffer = OrderedDict()
        split2stream = OrderedDict()
        split2writer = OrderedDict()
        split2imgid2row = {}
        split2currow = {}
        total = len(
            [
                p
                for p in Path(path).rglob(f"*.{img_format}")
                if dataset_name in str(p).lower()
            ]
        )
        for path in tqdm(Path(path).rglob(f"*.{img_format}"), total=total):
            if dataset_name in str(path).lower():
                split = path.parent.name

                if split not in split2buffer:
                    imgid2row = {}
                    cur_row = 0
                    # buffer = pyarrow.allocate_buffer(size=64, resizable=True)
                    buffer = pyarrow.BufferOutputStream()
                    # buffer = BytesIO()
                    split2buffer[split] = buffer
                    stream = pyarrow.output_stream(buffer)
                    split2stream[split] = stream
                    writer = ArrowWriter(features=features, stream=stream)
                    split2writer[split] = writer
                else:
                    imgid2row = split2imgid2row[split]
                    cur_row = split2currow[split]
                    buffer = split2buffer[split]
                    stream = split2stream[split]
                    writer = split2writer[split]

                # make sure file is not empty
                if path.stat().st_size < 10:
                    continue

                img_id = path.stem
                if img_id in imgid2row:
                    print(f"skipping {img_id}. Already written to table")
                imgid2row[img_id] = cur_row
                cur_row += 1
                split2currow[split] = cur_row
                split2imgid2row[split] = imgid2row
                filepath = str(path)

                if subset_ids is not None and img_id not in subset_ids:
                    continue

                # now do model forward
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
                assert set(features.keys()) == set(output_dict.keys()), (
                    f"mismatch between feature items"
                    f" and model output keys: {set(output_dict.keys()).symmetric_difference(set(features.keys()))}"
                )

                # write features
                batch = features.encode_batch(output_dict)
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
            # writer.finalize(close_stream=False)
            try:
                writer.finalize(close_stream=False)
            except Exception:
                pass

            # concat datasets if multiple splits
            # if len(dsets) != 1:
            #     dset = datasets.concatenate_datasets(dsets=dsets)

            # misc.
            dset = pickle.loads(pickle.dumps(dset))
            save_to = os.path.join(datadir, "arrow", dataset_name, f"{split}.arrow")
            os.makedirs(os.path.join(datadir, "arrow", dataset_name), exist_ok=True)

            # add extra metadata
            extra_meta = {"img_to_row_map": imgid2row}
            table = set_metadata(dset._data, tbl_meta=extra_meta)

            # define new writer
            writer = ArrowWriter(path=save_to, schema=table.schema, with_metadata=False)
            # raise Exception(writer._schema.metadata.keys())

            # save new table
            writer.write_table(table)
            e, b = Imageset.custom_finalize(writer, close_stream=True)
            print(f"Success! You wrote {e} entry(s) and {b >> 20} mb")
            print(f"Located: {save_to}")

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
    def from_file(cls, path, split=None):
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
    @abstractmethod
    def forward(filepath, image_preprocessor, model, **kwargs):
        raise Exception("child forward is not being called")

    @property
    @abstractmethod
    def name(self):
        return ""
