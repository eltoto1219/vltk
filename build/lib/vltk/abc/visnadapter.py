import json
import os
import pickle
from abc import abstractmethod
from collections import Counter

import datasets as ds
import pyarrow
import vltk
from datasets import ArrowWriter
from tqdm import tqdm
from vltk import ANNOTATION_DIR
from vltk.abc.adapter import Adapter
from vltk.inspection import collect_args_to_func
from vltk.processing.label import clean_imgid_default
from vltk.utils.base import set_metadata


class VisnDataset(Adapter):
    _batch_size = 1028
    _base_features = {
        vltk.imgid: ds.Value("string"),
    }
    _meta_names = {"img_to_row_map", "object_frequencies"}
    _is_annotation = True

    @classmethod
    def filepath(cls, imgid, datadir, split):
        filename = cls.imgid_to_filename(imgid, split)
        return os.path.join(datadir, cls.name, split, filename)

    @classmethod
    def load_imgid2path(cls, datadir, split):
        name = cls.__name__.lower()
        path = os.path.join(datadir, name, split)
        return VisnDataset.files(path)

    @staticmethod
    def files(path):
        files = {}
        if not os.path.isdir(path):
            print(f"No path exists for: {path}")
            return files
        for i in os.listdir(path):
            fp = os.path.join(path, i)
            iid = i.split(".")[0]
            files[iid] = fp
        return files

    @classmethod
    def extract(
        cls,
        searchdir,
        savedir=None,
        data_format="jpg",
        ignore_files=None,
        **kwargs,
    ):

        schema_dict = collect_args_to_func(cls.schema, kwargs=kwargs)
        feature_dict = {**cls.schema(**schema_dict), **cls._base_features}
        # lets work on doing the annotations first
        total_annos = {}
        searchdir, _ = cls._get_valid_search_pathes(
            searchdir, name=cls.__name__.lower(), annodir=ANNOTATION_DIR
        )
        files = cls._iter_files(searchdir)
        # get into right format
        json_files = []
        temp_splits = []
        print("loading annotations...")
        for anno_file in tqdm(files):
            if ignore_files is not None and ignore_files in str(anno_file):
                continue

            split = None
            for spl in vltk.SPLITALIASES:
                if spl in str(anno_file):
                    split = spl
                    break
            temp_splits.append(split)
            if "json" not in str(anno_file):
                continue
            if "caption" not in str(anno_file) and "question" not in str(anno_file):
                anno_data = json.load(open(str(anno_file)))
                json_files.append((str(anno_file), anno_data))

        forward_dict = collect_args_to_func(cls.forward, kwargs=kwargs)
        total_annos = cls.forward(json_files, temp_splits, **forward_dict)

        # now write
        print("writing to Datasets/Arrow object")
        writer, buffer, imgid2row, object_dict = cls._write_batches(
            total_annos, feature_dict, cls._batch_size
        )
        if savedir is None:
            savedir = searchdir

        extra_meta = {"img_to_row_map": imgid2row, "object_frequencies": object_dict}
        (table, meta_dict) = cls._write_data(writer, buffer, savedir, extra_meta)
        if table is None:
            return None
        return cls(arrow_table=table, meta_dict=meta_dict)

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
        # change feature types to classes isntead
        for i, entry in enumerate(annos):
            imgs_left = abs(i + 1 - n_files)
            # leave uncleaned actually
            img_id = entry[vltk.imgid]
            # for now, we will do a temporary fix
            if vltk.label in entry:
                object_dict.update(entry[vltk.label])
            else:
                for k, v in entry.items():
                    if isinstance(v, list) and all(
                        map(lambda x: isinstance(x, str), v)
                    ):
                        object_dict.update(v)
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
        value = buffer.getvalue()
        if value.size == 0:
            print("WARNING: no data saved")
            return (None, None)
            # do something
        dset = ds.Dataset.from_buffer(value)
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
        e, b = VisnDataset._custom_finalize(writer, close_stream=True)
        print(f"Success! You wrote {e} entry(s) and {b >> 20} mb")
        print(f"Located: {savefile}")
        return (table, extra_meta)

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

    @abstractmethod
    def forward(json_files, **kwargs):
        raise Exception("child forward is not being called")

    @abstractmethod
    def schema(*args, **kwargs):
        return dict

    # @abstractmethod
    # def imgid_to_filename(imgid, split):
    #     return str
