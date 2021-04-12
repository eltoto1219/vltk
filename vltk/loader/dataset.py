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
import numpy as np
import tokenizers
import torch
import vltk

# disable logging from datasets
from datasets.utils.logging import set_verbosity_error
from PIL import Image
from pycocotools import mask as Mask
from torch.utils.data import Dataset
from vltk.inspection import collect_args_to_func
from vltk.processing import data as data_proc
from vltk.processing.image import Pipeline

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

_data_procecessors = data_proc.Data()


def rescale_box(boxes, hw_scale):
    # boxes = (n, (x, y, w, h))
    # x = top left x position
    # y = top left y position
    h_scale = hw_scale[0]
    w_scale = hw_scale[1]
    y_centroids = (boxes[:, 1] - boxes[:, 3] / 2) * h_scale
    x_centroids = (boxes[:, 0] + boxes[:, 2] / 2) * w_scale
    boxes[:, 2] *= w_scale
    boxes[:, 3] *= h_scale
    boxes[:, 0] = x_centroids - boxes[:, 2] / 2  # scaled xs
    boxes[:, 1] = y_centroids + boxes[:, 3] / 2  # scaled ys
    return boxes


def seg_to_mask(segmentation, h, w):
    segmentation = Mask.decode(Mask.frPyObjects(segmentation, h, w))
    if len(segmentation.shape) == 3:
        segmentation = np.any(segmentation, axis=-1).astype(np.uint8)
    return segmentation


def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8) * 255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)


def uncompress_mask(compressed, size):
    mask = np.zeros(size, dtype=np.uint8)
    mask[compressed[0], compressed[1]] = 1
    return mask


class CollatedSets:
    def __init__(self, *args):

        self.args = args
        self.range2listpos = {}
        start = 0
        for i, a in enumerate(args):
            self.range2listpos[range(start, len(a) + start)] = i
            start += len(a)

    @property
    def _is_visndatasetadapter(self):
        if not hasattr(self, "__is_visndatasetadapter"):
            if all(map(lambda x: hasattr(x, "get"), self.args)):
                self.__is_visndatasetadapter = True
            else:
                self.__is_visndatasetadapter = False
            return self.__is_visndatasetadapter
        else:
            return self.__is_visndatasetadapter

    # TODO: figure out better solution if we start chaining idk lets say 10 datasets together
    def get(self, img_id):
        if not self._is_visndatasetadapter:
            raise Exception(
                "Only use this method if the datasets within this object are purely vision"
            )
        for visndatasetadapter in self.args:
            try:
                return visndatasetadapter.get(img_id)
            except KeyError:
                pass
        raise Exception("image id not found in any  visndatasetadapter annotations")

    def get_visnlangdatasetadapter_and_ind(self, x):
        if x >= len(self):
            raise IndexError(f"index {x} is out of range 0 to {len(self)}")
        for rng in self.range2listpos:
            if x in rng:
                listpos = self.range2listpos[rng]
                listind = x - rng.start
                return self.args[listpos], listind

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
        self.tokenizer = TOKENIZERS[config.tokenizer](VOCABPATH, lowercase=True)
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
        self.datasets = CollatedSets(*visnlangdatasetadapters)
        self.img2visnlangdatasetadapter = {}
        self.img2visdatasetadapter = {}
        self.uniq_imgs = set()
        # raise Exception("hello", self.visndatasetadapterdict)
        for is_info in self.visndatasetadapterdict.items():
            is_name, is_split_dict = is_info
            # raise Exception(is_name)
            for k, imgset in is_split_dict.items():
                # for split_name, imgset in self.visnlangdatasetadapterdict[is_name].items():
                if isinstance(imgset, dict):
                    img_ids = set(imgset.keys())
                else:
                    img_ids = set(imgset.imgids)
                for img_id in img_ids:
                    self.img2visdatasetadapter[img_id] = (is_name, k)
                self.uniq_imgs = self.uniq_imgs.union(img_ids)
                # print("HI", imgset._img_to_row_map)

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
    def image_transforms(self):
        return self._image_transforms

    @property
    def annotations(self):
        return getattr(self, "_annotations", None)

    def _handle_annotations(self, img_id):
        # TODO: I need to implement this function asap
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
                or k is vltk.image
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
        self._annotations = CollatedSets(*annotations)

    def _init_image_pipeline(self):
        config_dict = self.config.to_dict()
        func_dict = collect_args_to_func(Pipeline, config_dict)
        self._image_transforms = Pipeline(**func_dict)

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
        x.pop("img_id", None)
        x["text_attention_mask"] = encoded.attention_mask
        x["input_ids"] = encoded.ids
        x["type_ids"] = encoded.type_ids

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
        img_id = entry[vltk.imgid]
        proc_args = {"config": self.config}
        if self.config.rand_feats is not None:
            feat_shape = tuple(self.config.rand_feats)
            filepath = entry[vltk.image]
            entry[vltk.filepath] = filepath
            img = torch.rand(feat_shape)
            entry[vltk.image] = img

        elif self.config.extractor is None:
            filepath = entry[vltk.filepath]
            entry[vltk.filepath] = filepath
            entry[vltk.image] = self.image_transforms(filepath)
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
        # if self.annotationdict is not None:
        #     entry.update(self._handle_annotations(img_id))

        # now we do other image processors
        if self.config.image_processors is not None:
            for proc in self.config.image_processors:
                proc_func = _data_procecessors.get(proc)
                proc_func(entry, **proc_args)

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
        idxs = visnlangdatasetadapter.img_to_rows_map[img_id]
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
    def transpose_img2txt(batch, img_keys, device=None, max_size=36):
        if isinstance(device, list):
            device = device[0]
        # first we resize image according to how many examples that we need
        n_sents_per_img = [len(i) for i in batch["input_ids"]]
        for img_key in img_keys:
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
            if k not in img_keys:
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
        self._init_image_pipeline()
        self.visndatasetadapterdict = visndatasetadapterdict
        self.object_to_id = object_to_id
        self.img_id_to_path = {}
        self.n_imgs = 0
        # later if we need
        self.imgid_info = {}
        for imgsetsplits in list(visndatasetadapterdict.values()):
            for imgids2files in imgsetsplits.values():
                self.n_imgs += len(imgids2files)
                # for imgid, filepath in imgids2files.items():
                #    self.img_id_to_path
                self.img_id_to_path.update(imgids2files)

    @property
    def batch_size(self):
        if not self.is_train:
            return self.config.eval_batch_size
        else:
            return self.config.train_batch_size

    @property
    def image_transforms(self):
        return self._image_transforms

    @property
    def annotations(self):
        return self._annotations

    def update_objects(self, path_or_dict):
        if isinstance(path_or_dict, str):
            path_or_dict = json.load(open(path_or_dict))
        else:
            pass
        self.object_to_id = path_or_dict

    def _handle_annotations(self, img_id, img_info, img):
        skip_segmentation = False
        cur_size = img_info[vltk.size]
        if img_info is None:
            skip_segmentation = True
        else:
            if vltk.rawsize not in img_info:
                scale = (1.0, 1.0)
            else:
                raw_size = img_info[vltk.rawsize]
                scale = cur_size[0] / raw_size[0], cur_size[1] / raw_size[1]
        anno_dict = self.annotations.get(img_id)
        word_labels = anno_dict[vltk.label]
        labels = torch.Tensor([self.object_to_id[l] for l in word_labels])
        anno_dict[vltk.label] = labels
        for k, v in anno_dict.items():
            if (
                k is vltk.label
                or k is vltk.image
                or k == vltk.imgid
                or isinstance(v, torch.Tensor)
            ):
                continue
            elif k == vltk.segmentation and not skip_segmentation:
                size = anno_dict[vltk.size]
                try:
                    segmentations = np.stack(
                        [
                            resize_binary_mask(seg_to_mask(s, *size), cur_size)
                            for s in v
                            if s
                        ]
                    )
                except Exception:
                    raise Exception(v)
                values = torch.as_tensor(segmentations)

            else:
                if k == vltk.area or k == vltk.size:
                    values = torch.tensor(v)

                else:
                    values = list(map(lambda x: torch.Tensor(x), v))
                try:
                    values = torch.stack(values, dim=0)
                except Exception:
                    pass
                if k == vltk.box:
                    values = rescale_box(values, scale)

            anno_dict[k] = values

        if skip_segmentation and vltk.segmentation in anno_dict:
            anno_dict.pop(vltk.segmentation)

        return anno_dict

    def _init_annotation_dict(self, annotationdict):
        if annotationdict is None:
            raise Exception("HERE")
            self._annotations = None
            return
        annotations = list(annotationdict.values())
        self._annotations = CollatedSets(*annotations)

    def _init_image_pipeline(self):
        config_dict = self.config.to_dict()
        func_dict = collect_args_to_func(Pipeline, config_dict)
        self._image_transforms = Pipeline(**func_dict)

    def _handle_image(self, entry):
        img_id = entry[vltk.imgid]
        filepath = self.img_id_to_path[img_id]
        if self.config.rand_feats is not None:
            feat_shape = tuple(self.config.rand_feats)
            entry[vltk.filepath] = filepath
            img = torch.rand(feat_shape)
            entry[vltk.filepath] = img
            img_info = None
        else:
            entry[vltk.filepath] = filepath
            transformed = self.image_transforms(filepath)
            if isinstance(transformed, tuple):
                img, img_info = transformed
            else:
                img, img_info = transformed, {}
                size = transformed.shape[1:]
                img_info[vltk.size] = size
            for k, v in img_info.items():
                entry[k] = torch.Tensor(v)
            entry[vltk.image] = img

        if self.annotations is not None:
            entry.update(self._handle_annotations(img_id, img_info, img))

    def __len__(self):
        if self.annotations is not None:
            return len(self.annotations)
        else:
            return self.n_imgs

    @torch.no_grad()
    def __getitem__(self, i):
        anno_dict = self.annotations[i]
        self._handle_image(anno_dict)
        return anno_dict
