import inspect
# note if we do not immport a pacakage correctly in this class, no loops or exps will be present
import json
import os
import resource
import sys

import torch
import vltk
from datasets.utils.logging import set_verbosity_error
from torch.nn.utils.rnn import pad_sequence
# disable logging from datasets
from vltk.loader.basedataset import BaseDataset, CollatedVisionSets
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


class VisionDataset(BaseDataset):
    _supported = (vltk.polygons, vltk.size, vltk.area, vltk.box, vltk.RLE)

    def __init__(
        self,
        config,
        visndatasetadapterdict,
        annotationdict=None,
        object_to_id=None,
        is_train=False,
        all_same_keys=True,
    ):
        self.is_train = is_train
        # self.annotationdict = annotationdict
        self._init_annotation_dict(config, annotationdict)
        self.config = config
        self._init_image_processor(config)
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
        self.all_same_keys = all_same_keys

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

    def _init_annotation_dict(self, config, annotationdict):
        if annotationdict is None or config.ignore_annotations:
            self._annotations = None
        else:
            annotations = list(annotationdict.values())
            self._annotations = CollatedVisionSets(*annotations)

    def _init_image_processor(self, config):
        if config.extractor is None:
            processor = config.visn.build()
            self._image = processor
            self._transforms = self._image.transforms

    @property
    def transforms(self):
        return {t.__class__.__name__: t for t in self._transforms}

    def _handle_image(self, entry):
        img_id = entry[vltk.imgid]
        if vltk.filepath not in entry:
            filepath = self.img_id_to_path[img_id]
            entry[vltk.filepath] = filepath
        else:
            filepath = entry[vltk.filepath]
        if self.config.rand_feats is not None:
            feat_shape = tuple(self.config.rand_feats)
            img = torch.rand(feat_shape)
            entry[vltk.img] = img
        else:
            if not self.config.ignore_filepath:
                entry[vltk.filepath] = filepath
            else:
                entry.pop(vltk.filepath)
            entry[vltk.img] = self.image(filepath)

        entry[vltk.size] = get_size(self.image)
        entry[vltk.rawsize] = get_rawsize(self.image)
        if torch.all(entry[vltk.size].eq(entry[vltk.rawsize])):
            entry.pop(vltk.rawsize)
        else:
            entry[vltk.scale] = get_scale(self.image)
        return entry

    def _handle_annotations(self, entry):
        img_id = entry[vltk.imgid]
        skip_segmentation = (
            True
            if vltk.size not in entry and not self.config.ignore_segmentation
            else False
        )
        # get annotations for image
        entry.update(self.annotations.get(img_id))
        if skip_segmentation and vltk.polygons in entry:
            entry.pop(vltk.polygons)
        if skip_segmentation and vltk.RLE in entry:
            entry.pop(vltk.RLE)
        # TODO: need better solution for later, but now were dumping all string labels
        # into the object to id dictionary
        # add annotation labels to image
        if vltk.label in entry:
            word_labels = entry[vltk.label]
            labels = torch.Tensor(self.answer_to_id[word_labels])
            entry[vltk.label] = labels
        if vltk.objects in entry:
            word_labels = entry[vltk.objects]
            if len(word_labels) < self.config.max_objects:
                n_objects = len(word_labels)
                entry[vltk.n_objects] = torch.Tensor(n_objects)
                word_labels += [""] * (self.config.max_objects - n_objects)
            else:
                word_labels = word_labels[: self.config.max_objects]
            labels = torch.Tensor([self.object_to_id[l] for l in word_labels])
            entry[vltk.objects] = labels

        # we go through user-defined annoations first
        for k, v in entry.items():
            if k not in vltk.SUPPORTEDNAMES and k != vltk.objects:
                # take care of lists of strings
                try:
                    prim = base.get_list_primitive(v)
                except Exception:
                    raise Exception(k, v)
                if prim == str:
                    values = base.convertids_recursive(v, self.object_to_id)
                    entry[k] = values

        # only loop through annotations processed by vltk
        for k in self._supported:
            if k not in entry:
                continue
            v = entry[k]
            if k == vltk.polygons and not skip_segmentation:
                size = entry[vltk.size]
                if vltk.rawsize not in entry:
                    rawsize = size
                else:
                    rawsize = entry[vltk.rawsize]
                segs = torch.stack(
                    list(
                        map(
                            lambda x: resize_binary_mask(
                                seg_to_mask(x, *rawsize), size
                            ),
                            entry.pop(k),
                        ),
                    )
                )

                segs = segs[: min(len(segs), self.config.max_objects)]
                segs = torch.nn.functional.pad(
                    segs,
                    (0, 0, 0, 0, 0, self.config.max_objects - len(segs)),
                )
                entry[vltk.segmentation] = segs
            elif k == vltk.RLE and not skip_segmentation:

                # s = time.time()
                segs = torch.stack(
                    list(
                        map(
                            lambda x: resize_binary_mask(
                                imagepoints_to_mask(x, entry[vltk.rawsize]),
                                torch.as_tensor(entry[vltk.size]),
                            ),
                            entry.pop(k),
                        )
                    )
                )
                segs = segs[: min(len(segs), self.config.max_objects)]
                segs = torch.nn.functional.pad(
                    segs,
                    (0, 0, 0, 0, 0, self.config.max_objects - len(segs)),
                )
                entry[vltk.segmentation] = segs

            elif k == vltk.box:
                values = torch.tensor(v)
                values = values[: min(len(values), self.config.max_objects)]

                if vltk.scale in entry:
                    values = rescale_box(values, entry[vltk.scale])
                values = torch.nn.functional.pad(
                    values,
                    (0, 0, 0, self.config.max_objects - len(values)),
                )
                entry[k] = values

        return entry

    def __len__(self):
        return self.n_imgs

    @torch.no_grad()
    def __getitem__(self, i):
        if len(self.imgids) == len(self.annotations):
            anno_dict, anno_dataset = self.annotations[i]
        else:
            anno_dict = self.annotations.get(self.imgids[i])
        anno_dict = self._handle_image(anno_dict)
        if self.annotations is not None:
            anno_dict = self._handle_annotations(anno_dict)
        return anno_dict
        # if not self.all_same_keys:
        #     return anno_dict
        # else:
        #     return {anno_dataset: anno_dict}
