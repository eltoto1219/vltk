import inspect
import json
import logging as logger
import os
import pickle
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
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
from vltk.processing.image import Pipeline
from vltk.utils import set_metadata

__all__ = ["Extractor", "Extractors"]

_imageproc = image_proc.Image()

ANNOTATION_DIR = "boxes"
DEFAULT_ANNOS = {
    "bbox": ds.Sequence(
        length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("float32"))
    ),
    "segmentation": ds.Sequence(
        length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("float32"))
    ),
    "area": ds.Sequence(length=-1, feature=ds.Value("float32")),
    IMAGEKEY: ds.Value("string"),
}


class Extractors:
    def __init__(self):
        if "IMAGESETDICT" not in globals():
            global IMAGESETDICT
            IMAGESETDICT = get_classes(IMAGESETPATH, ds.Dataset, pkg="vltk.extractor")

    def avail(self):
        return list(IMAGESETDICT.keys())

    def get(self, name):
        return IMAGESETDICT[name]

    def add(self, name, dset):
        IMAGESETDICT[name] = dset


class Extractor(ds.Dataset, ABC):
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

    @staticmethod
    def get_valid_search_pathes(searchdirs, name, splits, ignore_splits=False):
        if isinstance(splits, str):
            splits = [splits]
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
            if not ignore_splits:
                for s in splits:
                    p2 = os.path.join(p1, s)
                    if not os.path.isdir(p2):
                        continue
                    else:
                        new_searchdirs.append(p2)
            else:
                new_searchdirs.append(p1)

        assert new_searchdirs
        return new_searchdirs

    @staticmethod
    def iter_files(searchdirs, img_format=None):
        for s in searchdirs:
            for f in os.listdir(s):
                if img_format is not None:
                    if img_format in f:
                        file = Path(os.path.join(s, f))
                        if file.stat().st_size > 10:
                            yield file
                else:
                    file = Path(os.path.join(s, f))
                    if file.stat().st_size > 10:
                        yield Path(os.path.join(s, f))

    # multiple things that we can do with this mehtod, we can just extract other random data
    # for each extractor
    @classmethod
    def extract(
        cls,
        name,
        img_config,
        splits,
        searchdirs,
        model=None,
        features=None,
        savedir=os.getcwd(),
        img_format="jpg",
        subset_ids=None,
        annotations=True,
        annotation_key=None,
        **kwargs,
    ):
        if model is None and not annotations:
            print(
                "WARNING: there is not model for feature extraction and no annotations to save"
            )

        # lets work on doing the annotations first
        if annotations:
            print("extracting annotations")
            anno_schema = ds.Features(DEFAULT_ANNOS)
            total_annos = {}
            searchdirs = cls.get_valid_search_pathes(
                searchdirs, name, ANNOTATION_DIR, ignore_splits=False
            )
            files = cls.iter_files(searchdirs, img_format="json")
            # get into right format
            for anno_file in files:
                if "instance" in str(anno_file):
                    if annotation_key is not None:
                        anno_data = json.load(open(str(anno_file)))[annotation_key]
                    else:
                        anno_data = json.load(open(anno_file))["annotations"]
                for entry in tqdm(anno_data):
                    img_id = Label.clean_imgid_default(str(entry["image_id"]))

                    bbox = entry["bbox"]
                    area = entry["area"]
                    segmentation = entry["segmentation"]

                    img_data = total_annos.get(img_id, None)
                    if img_data is None:
                        img_entry = defaultdict(list)
                        img_entry["bbox"].append(bbox)
                        img_entry["area"].append(area)
                        for s in segmentation:
                            assert not isinstance(s[0], list)
                            if all(map(lambda x: isinstance(x, float), s)):
                                img_entry["segmentation"].append(s)

                        total_annos[img_id] = img_entry
                    else:
                        total_annos[img_id]["bbox"].append(bbox)
                        total_annos[img_id]["area"].append(area)
                        for s in segmentation:
                            assert not isinstance(s[0], list)

                            if all(map(lambda x: isinstance(x, float), s)):
                                total_annos[img_id]["segmentation"].append(s)

                break

            # now write
            batch_size = cls._batch_size
            imgid2row = {}
            cur_size = 0
            cur_row = 0
            cur_batch = None
            buffer = pyarrow.BufferOutputStream()
            stream = pyarrow.output_stream(buffer)
            writer = ArrowWriter(features=anno_schema, stream=stream)
            total_files = len(total_annos)
            raise Exception(len(total_annos))
            for i, (img_id, entry) in enumerate(total_annos.items()):
                imgs_left = abs(i + 1 - total_files)
                entry["img_id"] = img_id
                if img_id in imgid2row:
                    print(f"skipping {img_id}. Already written to table")
                imgid2row[img_id] = cur_row
                cur_row += 1
                if cur_size == 0:
                    # entry["img_id"] = [entry["img_id"]]
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
                    batch = anno_schema.encode_batch(cur_batch)
                    # try:
                    # except Exception:
                    #     raise Exception(cur_batch)
                    writer.write_batch(batch)
                    cur_batch = None

            # define datasets
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
            extra_meta = {"img_to_row_map": imgid2row}
            table = set_metadata(dset._data, tbl_meta=extra_meta)
            # define new writer
            writer = ArrowWriter(
                path=savefile, schema=table.schema, with_metadata=False
            )
            # savedir new table
            writer.write_table(table)
            e, b = Extractor.custom_finalize(writer, close_stream=True)
            print(f"Success! You wrote {e} entry(s) and {b >> 20} mb")
            print(f"Located: {savefile}")

        # now lets try feature extraction
        if model is not None:
            print("extracting features")
            searchdirs = cls.get_valid_search_pathes(searchdirs, name, splits)
            extractor_name = cls.name
            savepath = os.path.join(savedir, name, extractor_name)
            print(f"will write to {savepath}")
            os.makedirs(savepath, exist_ok=True)
            img_processor_args = collect_args_to_func(Pipeline, img_config.to_dict())
            img_preprocessor = Pipeline(**img_processor_args)
            # prelim forward checks
            cls._check_forward(img_preprocessor, model, cls.forward)
            # ensure features are in correct format
            features = cls._check_features(features, cls.default_features, kwargs)
            # setup tracking dicts
            split2buffer = OrderedDict()
            split2stream = OrderedDict()
            split2writer = OrderedDict()
            split2imgid2row = {}
            split2currow = {}
            # begin search
            print(f"extracting from in {searchdirs}")
            batch_size = cls._batch_size
            cur_size = 0
            cur_batch = None
            files = cls.iter_files(searchdirs, img_format)
            total_files = len(set(files))
            for i, path in tqdm(
                enumerate(cls.iter_files(searchdirs, img_format)), file=sys.stdout
            ):
                split = path.parent.name
                img_id = path.stem
                img_id = Label.clean_imgid_default(img_id)
                imgs_left = abs(i + 1 - total_files)
                if splits is not None and split not in splits:
                    continue
                if subset_ids is not None and img_id not in subset_ids:
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
                output_dict = cls.forward(
                    filepath=filepath,
                    image_preprocessor=img_preprocessor,
                    model=model,
                    **kwargs,
                )
                assert isinstance(
                    output_dict, dict
                ), "model outputs should be in dict format"
                output_dict["img_id"] = [img_id]

                if cur_batch is None or cur_size == 0:
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
                    batch = features.encode_batch(cur_batch)
                    writer.write_batch(batch)
                split2imgid2row[split] = imgid2row

            # define datasets
            dsets = []
            splitdict = {}
            print("saving...")
            for (_, writer), (split, b) in zip(
                split2writer.items(), split2buffer.items()
            ):
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
                e, b = Extractor.custom_finalize(writer, close_stream=True)
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
            feature_dict[IMAGEKEY] = Extractor._base_features[IMAGEKEY]
            features = ds.Features(feature_dict)
        elif isinstance(features, dict):
            features[IMAGEKEY] = Extractor._base_features[IMAGEKEY]
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
