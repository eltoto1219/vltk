import inspect
# note if we do not immport a pacakage correctly in this class, no loops or exps will be present
import json
import math
import os
import random
import resource
import sys
from collections.abc import Iterable
from copy import deepcopy

import numpy
import torch
import vltk
# disable logging from datasets
from datasets.utils.logging import set_verbosity_error
from torch.utils.data import Dataset
from vltk.utils import base
from vltk.utils.adapters import (Data, get_rawsize, get_scale, get_size,
                                 imagepoints_to_mask, rescale_box,
                                 resize_binary_mask, seg_to_mask)

__import__("tokenizers")
TOKENIZERS = {
    m[0]: m[1] for m in inspect.getmembers(sys.modules["tokenizers"], inspect.isclass)
}

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (6144, rlimit[1]))

set_verbosity_error()

VOCABPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "libdata/bert-base-uncased-vocab.txt")
).replace("loader/", "")
TOKENIZEDKEY = "encoded"
global TORCHCOLS
TORCHCOLS = set()
os.environ["TOKENIZERS_PARALLELISM"] = "False"

_data_procecessors = Data()


class CollatedVLSets:
    def __init__(self, *args):

        self.args = args
        self.range2listpos = {}
        start = 0
        for i, a in enumerate(args):
            self.range2listpos[range(start, len(a) + start)] = i
            start += len(a)

    # TODO: better solution for more datasets
    def get(self, img_id):
        for adapter in self.args:
            try:
                return adapter.get(img_id)
            except KeyError:
                pass
        raise Exception("image id not found in any  visndatasetadapter annotations")

    def __getitem__(self, x):
        if x >= len(self):
            raise IndexError(f"index {x} is out of range 0 to {len(self)}")
        for rng in self.range2listpos:
            if x in rng:
                listpos = self.range2listpos[rng]
                listind = x - rng.start
                return self.args[listpos][listind]

    def __len__(self):
        return sum(map(lambda x: len(x), self.args))

    def __iter__(self):
        return iter(map(lambda x: self[x], range(0, len(self))))


class CollatedVisionSets(CollatedVLSets):
    pass


class VisionLanguageDataset(Dataset):
    def __init__(
        self,
        config,
        visnlangdatasetadapterdict,
        visndatasetadapterdict,
        annotationdict=None,
        answer_to_id=None,
        object_to_id=None,
        is_train=False,
    ):
        self.annotationdict = annotationdict
        self.config = config
        self.is_train = is_train
        try:
            self.tokenizer = TOKENIZERS[config.tokenizer](VOCABPATH, lowercase=True)
        except KeyError:
            raise Exception(
                f"{config.tokenizer} not available. Try one of: {TOKENIZERS.keys()}"
            )
        self.tokenizer.add_special_tokens(self.special_tokens)
        self.tokenizer.enable_truncation(max_length=config.sent_length)
        self.tokenizer.enable_padding(length=config.sent_length)
        self._init_image_pipeline()
        splits = set()
        for v in visnlangdatasetadapterdict.values():
            splits = splits.union(set(v.keys()))
        self.splits = splits
        self.visnlangdatasetadapters = []
        self.visnlangdatasetadapterdict = visnlangdatasetadapterdict
        self.visndatasetadapterdict = visndatasetadapterdict
        self.answer_to_id = answer_to_id
        self.object_to_id = object_to_id
        self.uniq_labels = set(answer_to_id.keys())
        visnlangdatasetadapters = []
        # map special function and create list of visnlangdatasetadapters
        for dset in self.visnlangdatasetadapterdict:
            for split in self.visnlangdatasetadapterdict[dset]:
                visnlangdatasetadapter = self.visnlangdatasetadapterdict[dset][split]
                visnlangdatasetadapters.append(visnlangdatasetadapter)
                self.visnlangdatasetadapterdict[dset][split] = visnlangdatasetadapter
        self.datasets = CollatedVLSets(*visnlangdatasetadapters)
        self.img2visnlangdatasetadapter = {}
        self.img2visdatasetadapter = {}
        self.uniq_imgs = set()
        # raise Exception("hello", self.visndatasetadapterdict)
        for is_info in self.visndatasetadapterdict.items():
            is_name, is_split_dict = is_info
            # raise Exception(is_name)
            for k, imgset in is_split_dict.items():
                try:
                    img_ids = set(imgset.imgids)
                except Exception:
                    img_ids = set(imgset.keys())

                for img_id in img_ids:
                    self.img2visdatasetadapter[img_id] = (is_name, k)
                self.uniq_imgs = self.uniq_imgs.union(img_ids)

        all_ts_imgs = set()
        for ts_name, ts_splits in self.visnlangdatasetadapterdict.items():
            for split_name, ts in self.visnlangdatasetadapterdict[ts_name].items():
                temp_uniq = ts.uniq_imgs
                temp_uniq = self.uniq_imgs.intersection(temp_uniq)
                all_ts_imgs = all_ts_imgs.union(temp_uniq)
                # TODO: change this later
                for img in temp_uniq:
                    self.img2visnlangdatasetadapter[img] = (ts_name, split_name)

        self.uniq_imgs = tuple(all_ts_imgs)
        special_ids = set([self.tokenizer.token_to_id(t) for t in self.special_tokens])
        self.special_ids = deepcopy(special_ids)
        all_ids = [i[1] for i in self.tokenizer.get_vocab().items()]
        self.all_ids = deepcopy(all_ids)

    def update_labels(self, path_or_dict):
        if isinstance(path_or_dict, str):
            path_or_dict = json.load(open(path_or_dict))
        else:
            pass
        self.answer_to_id = path_or_dict
        self.uniq_labels = set(path_or_dict.keys())

    def update_objects(self, path_or_dict):
        if isinstance(path_or_dict, str):
            path_or_dict = json.load(open(path_or_dict))
        else:
            pass
        self.object_to_id = path_or_dict

    @property
    def image(self):
        return getattr(self, "_image", None)

    def _init_image_processor(self):
        processor = self.config.image.build()
        self._image = processor

    @property
    def annotations(self):
        return getattr(self, "_annotations", None)

    def _handle_annotations(self, img_id):
        annotation_pointer = self.annotations
        if annotation_pointer is None:
            raise Exception
            return
        anno_dict = annotation_pointer.get(img_id)
        labels = anno_dict[vltk.label]
        labels = torch.Tensor([self.object_to_id[int(l)] for l in labels])
        anno_dict[vltk.label] = labels
        for k, v in anno_dict.items():
            if (
                k is vltk.label
                or k is vltk.img
                or (isinstance(v, Iterable) and not isinstance(v[0], list))
            ):
                continue

            anno_dict[k] = list(map(lambda x: torch.Tensor(x), v))

        return anno_dict

    def _init_annotation_dict(self, annotationdict):
        if annotationdict is None:
            self._annotations = None
            return
        annotations = list(annotationdict.values())
        raise Exception("HERE", annotations)
        self._annotations = CollatedVLSets(*annotations)

    def processor_args(self):
        max_rand_sents = 1 if not self.config.img_first else 32
        return {
            "tokenizer": self.tokenizer,
            "config": self.config,
            "random_sents": [self.random_sent() for i in range(max_rand_sents)],
            "special_ids": self.special_ids,
            "answer_to_id": self.answer_to_id,
            "all_ids": self.all_ids,
            "n_ids": len(self.all_ids),
        }

    @staticmethod
    def text_map_function(x, proc_args):
        config = proc_args.get("config")
        tokenizer = proc_args.get("tokenizer")
        answer_to_id = proc_args.get("answer_to_id")
        text_processors = config.text_processors
        if text_processors is not None:
            if "matched_sentence_modeling" in text_processors:
                proc_func = _data_procecessors.get("matched_sentence_modeling")
                x = proc_func(x, **proc_args)
                TORCHCOLS.add("is_matched")

        encoded = tokenizer.encode(x.pop(vltk.text))
        x.pop(vltk.imgid, None)
        x[vltk.text_attention_mask] = encoded.attention_mask
        x[vltk.input_ids] = encoded.ids
        x[vltk.type_ids] = encoded.type_ids

        if vltk.label in x:
            label = x.pop(vltk.label)
            if label != config.ignore_id:
                lids = []
                for l in label:
                    lid = answer_to_id[l]
                    lids.append(lid)
                x[vltk.label] = lids

        # now we do other text processors
        if text_processors is not None:
            for proc in text_processors:
                if proc == "matched_sentence_modeling":
                    continue
                proc_func = _data_procecessors.get(proc)
                proc_func(x, **proc_args)

        # now we do label proccesor
        if config.label_processor is not None:
            proc_func = _data_procecessors.get(config.label_processor)
            proc_func(x, **proc_args)

        for k, v in x.items():
            if isinstance(v, list) and not isinstance(v[0], str):
                TORCHCOLS.add(k)
        if vltk.label in x:
            TORCHCOLS.add(vltk.label)

        return x

    def __len__(self):
        if self.config.img_first:
            return int(math.floor(len(self.uniq_imgs) * self.config.percent))
        else:
            return int(math.floor(len(self.datasets) * self.config.percent))

    @property
    def special_tokens(self):
        return [
            "[unk]",
            "[sep]",
            "[pad]",
            "[cls]",
            "[mask]",
        ]

    def _handle_image(self, entry):
        proc_args = {"config": self.config}
        if self.config.rand_feats is not None:
            feat_shape = tuple(self.config.rand_feats)
            filepath = entry[vltk.img]
            entry[vltk.filepath] = filepath
            img = torch.rand(feat_shape)
            entry[vltk.img] = img
        elif self.config.extractor is None:
            filepath = entry[vltk.filepath]
            entry[vltk.filepath] = filepath
            entry[vltk.img] = self.image(filepath)
        else:
            for k, v in entry.items():
                if isinstance(v, Iterable):
                    if isinstance(v, numpy.ndarray):
                        entry[k] = torch.from_numpy(v)
                    elif isinstance(v, list):
                        entry[k] = torch.tensor(v)
                elif isinstance(v, int) or isinstance(v, float):
                    entry[k] = torch.tensor(v)
            # TODO: okay I defintely need to change this
            if vltk.features in entry:
                proc_args["random_feat_func"] = self.random_feat
                entry[vltk.features] = entry[vltk.features][: self.config.max_objects]
        if self.annotationdict is not None:
            entry.update(self._handle_annotations(entry[vltk.imgid]))

    def random_feat(self):
        rand_ind = random.randint(0, len(self.uniq_imgs) - 1)
        img_id = self.uniq_imgs[rand_ind]
        ts_name, ts_split = self.img2visnlangdatasetadapter[img_id]
        visnlangdatasetadapter = self.visnlangdatasetadapterdict[ts_name][ts_split]
        is_name, is_split = zip(*visnlangdatasetadapter.data_info[ts_split].items())
        visndatasetadapter = self.visndatasetadapterdict[is_name[0]][is_split[0][0]]
        img_info = visndatasetadapter.get(img_id)
        if vltk.features in img_info:
            feat = random.choice(img_info[vltk.features])
            return feat
        else:
            return None

    def random_sent(self):
        rand_ind = random.randint(0, len(self.datasets) - 1)
        text_info = self.datasets[rand_ind]
        rand_sent = text_info[vltk.text]
        return rand_sent

    def _map(self, small_visnlangdatasetadapter):
        proc_args = self.processor_args()
        return small_visnlangdatasetadapter.map(
            lambda x: VisionLanguageDataset.text_map_function(x, proc_args=proc_args)
        )

    def _do_map_img_first(self, i):
        img_id = self.uniq_imgs[i]
        ts_name, ts_split = self.img2visnlangdatasetadapter[img_id]
        visnlangdatasetadapter = self.visnlangdatasetadapterdict[ts_name][ts_split]
        idxs = visnlangdatasetadapter.img_to_row_map[img_id]
        small_visnlangdatasetadapter = visnlangdatasetadapter.select(idxs)
        img_text_dict = self._map(small_visnlangdatasetadapter)

        img_text_dict.set_format(
            type="torch", output_all_columns=True, columns=list(TORCHCOLS)
        )
        # so what we have to turn into tensors ar
        img_text_dict = img_text_dict[:]
        img_text_dict[vltk.features] = img_id
        return img_text_dict, visnlangdatasetadapter, ts_split, img_id

    def _do_map_text_first(self, i):
        visnlangdatasetadapter, ind = self.datasets.get_visnlangdatasetadapter_and_ind(
            i
        )
        small_visnlangdatasetadapter = visnlangdatasetadapter[ind]
        img_id = small_visnlangdatasetadapter[vltk.imgid]
        proc_args = self.processor_args()
        text_info = self.text_map_function(small_visnlangdatasetadapter, proc_args)
        text_info = dict(
            map(
                lambda x: (
                    x[0],
                    torch.tensor(x[1]) if x[0] in TORCHCOLS else x[1],
                ),
                text_info.items(),
            )
        )

        return text_info, img_id, visnlangdatasetadapter

    @torch.no_grad()
    def __getitem__(self, i):
        if self.config.img_first:
            (
                img_text_dict,
                visnlangdatasetadapter,
                ts_split,
                img_id,
            ) = self._do_map_img_first(i)
            # is_name, is_split = zip(*visnlangdatasetadapter.data_info[ts_split].items())
            is_name, is_split = self.img2visdatasetadapter[img_id]
            visndatasetadapter = self.visndatasetadapterdict[is_name][is_split]
            img_info_dict = visndatasetadapter.get(img_id)
            if isinstance(img_info_dict, str):
                img_info_dict = {vltk.filepath: img_info_dict, vltk.imgid: img_id}
            self._handle_image(img_info_dict)
            entry = {**img_text_dict, **img_info_dict}
            # entry = img_info_dict
            return entry
        else:

            text_info, img_id, visnlangdatasetadapter = self._do_map_text_first(i)
            ts_name, ts_split = self.img2visnlangdatasetadapter[img_id]
            is_name, is_split = zip(*visnlangdatasetadapter.data_info[ts_split].items())
            visndatasetadapter = self.visndatasetadapterdict[is_name[0]][is_split[0][0]]
            img_info_dict = visndatasetadapter.get(img_id)
            if isinstance(img_info_dict, str):
                img_info_dict = {vltk.filepath: img_info_dict, vltk.imgid: img_id}
            self._handle_image(img_info_dict)
            entry = {**text_info, **img_info_dict}

            return entry

    @property
    def batch_size(self):
        if not self.is_train:
            return self.config.eval_batch_size
        else:
            return self.config.train_batch_size

    @staticmethod
    # TODO: unfinished
    def flatten_text(batch, flatten_keys=None):
        raise Exception
        if flatten_keys is None:
            flatten_keys = {"input_ids", "type_ids", "text_attention_mask", "label"}
        for f in flatten_keys:
            flattened = None
            key = batch[f]
            for i in key:
                if flattened is None:
                    key[i] = flattened
                else:
                    flattened = torch.cat((flattened, key[i]), dim=0)
            batch[f] = flattened

    @staticmethod
    def transpose(batch, device=None, max_size=36):
        if isinstance(device, list):
            device = device[0]
        # first we resize image according to how many examples that we need
        n_sents_per_img = [len(i) for i in batch["input_ids"]]
        for img_key, v in batch.keys():
            if not isinstance(v, list):
                assert img_key in batch, f"{img_key} not in {list(batch.keys())}"
                imgs = torch.cat(
                    [
                        i.unsqueeze(0).expand(min(n, max_size), *i.shape)
                        for i, n in zip(batch.pop(img_key), n_sents_per_img)
                    ],
                    dim=0,
                )
                batch[img_key] = imgs
                if device is not None:
                    batch[img_key] = batch[img_key].to(device)
        # then we convert the other things in the dataset to torch tensors if we can
        for k in batch:
            if isinstance(v, list):
                if isinstance(batch[k][0], torch.Tensor):
                    batch[k] = torch.cat(
                        [
                            j[: min(max_size, n)]
                            for i, (j, n) in enumerate(zip(batch[k], n_sents_per_img))
                        ],
                        dim=0,
                    )
                    if device is not None:
                        batch[k] = batch[k].to(device)
                elif isinstance(batch[k][0], str):
                    new_v = []
                    # here is also a part that we want to recude
                    for i, n in zip(batch[k], n_sents_per_img):
                        if n >= max_size:
                            n = min(n, max_size)
                        new_v.extend(i * n)
                    batch[k] = new_v


class VisionDataset(Dataset):
    _supported = (vltk.polygons, vltk.size, vltk.area, vltk.box, vltk.points)

    def __init__(
        self,
        config,
        visndatasetadapterdict,
        annotationdict=None,
        object_to_id=None,
        is_train=False,
    ):
        self.is_train = is_train
        # self.annotationdict = annotationdict
        self._init_annotation_dict(annotationdict)
        self.config = config
        self._init_image_processor()
        self.visndatasetadapterdict = visndatasetadapterdict
        self.object_to_id = object_to_id
        self.img_id_to_path = {}
        self.n_imgs = 0
        # later if we need
        self.idx_to_imgid = {}
        for imgsetsplits in list(visndatasetadapterdict.values()):
            for imgids2files in imgsetsplits.values():
                self.n_imgs += len(imgids2files)
                self.img_id_to_path.update(imgids2files)
        self.imgids = tuple(self.img_id_to_path.keys())

    @property
    def batch_size(self):
        if not self.is_train:
            return self.config.eval_batch_size
        else:
            return self.config.train_batch_size

    @property
    def image(self):
        return self._image

    @property
    def annotations(self):
        return self._annotations

    def update_objects(self, path_or_dict):
        if isinstance(path_or_dict, str):
            path_or_dict = json.load(open(path_or_dict))
        else:
            pass
        self.object_to_id = path_or_dict

    def _init_annotation_dict(self, annotationdict):
        if annotationdict is None:
            self._annotations = None
        else:
            annotations = list(annotationdict.values())
            self._annotations = CollatedVisionSets(*annotations)

    def _init_image_processor(self):
        if self.config.extractor is None:
            processor = self.config.image.build()
            self._image = processor
            self._transforms = self._image.transforms

    @property
    def transforms(self):
        return {t.__class__.__name__: t for t in self._transforms}

    def _handle_image(self, entry):
        img_id = entry[vltk.imgid]
        filepath = self.img_id_to_path[img_id]
        entry[vltk.filepath] = filepath
        if self.config.rand_feats is not None:
            feat_shape = tuple(self.config.rand_feats)
            img = torch.rand(feat_shape)
            entry[vltk.img] = img
        else:
            entry[vltk.filepath] = filepath
            entry[vltk.img] = self.image(filepath)

        entry[vltk.size] = get_size(self.image)
        entry[vltk.scale] = get_scale(self.image)
        entry[vltk.rawsize] = get_rawsize(self.image)

    def _handle_annotations(self, entry):
        img_id = entry[vltk.imgid]
        skip_segmentation = True if vltk.size not in entry else False
        # get annotations for image
        entry.update(self.annotations.get(img_id))
        if skip_segmentation and vltk.polygons in entry:
            entry.pop(vltk.polygons)
        if skip_segmentation and vltk.points in entry:
            entry.pop(vltk.points)
        # TODO: need better solution for later, but now were dumping all string labels
        # into the object to id dictionary
        # add annotation labels to image
        if vltk.label in entry:
            word_labels = entry[vltk.label]
            labels = torch.Tensor([self.object_to_id[l] for l in word_labels])
            entry[vltk.label] = labels

        # we go through user-defined annoations first
        for k, v in entry.items():
            if k not in vltk.SUPPORTEDNAMES:
                # take care of lists of strings
                prim = base.get_list_primitive(v)
                if prim == str:
                    values = base.convertids_recursive(v, self.object_to_id)
                    entry[k] = values

        # only loop through annotations processed by vltk
        for k in self._supported:
            if k not in entry:
                continue
            v = entry[k]
            if k == vltk.polygons and not skip_segmentation:
                entry[vltk.segmentation] = torch.tensor(
                    list(
                        map(
                            lambda x: resize_binary_mask(
                                seg_to_mask(x, *entry[vltk.rawsize]), entry[vltk.size]
                            ),
                            v,
                        ),
                    )
                )
                entry.pop(k)
            elif k == vltk.points:

                # s = time.time()
                entry[vltk.segmentation] = torch.stack(
                    list(
                        map(
                            lambda x: resize_binary_mask(
                                imagepoints_to_mask(x, entry[vltk.rawsize]),
                                torch.as_tensor(entry[vltk.size]),
                            ),
                            v,
                        )
                    )
                )
                # print(time.time() - s)
                entry.pop(k)
                # raise Exception(entry[vltk.img].shape)

            elif k == vltk.box:
                values = torch.tensor(v)
                values = rescale_box(values, entry[vltk.scale])
                entry[k] = values

        return entry

    def __len__(self):
        return self.n_imgs

    @torch.no_grad()
    def __getitem__(self, i):
        if len(self.imgids) == len(self.annotations):
            anno_dict = self.annotations[i]
        else:
            anno_dict = self.annotations.get(self.imgids[i])
        self._handle_image(anno_dict)
        if self.annotations is not None:
            self._handle_annotations(anno_dict)
        return anno_dict
