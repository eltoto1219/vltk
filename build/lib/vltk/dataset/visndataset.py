import inspect
# note if we do not immport a pacakage correctly in this class, no loops or exps will be present
import json
import os
import resource
import sys
from collections import defaultdict
from itertools import chain

import torch
import vltk.vars as vltk
from datasets.utils.logging import set_verbosity_error
# disable logging from datasets
from vltk.dataset.basedataset import BaseDataset, CollatedVisionSets
from vltk.processing import Processors, VisnProcessor
from vltk.utils import base
from vltk.utils.adapters import (get_rawsize, get_scale, get_size,
                                 imagepoints_to_mask, rescale_box,
                                 resize_binary_mask, seg_to_mask)

__import__("tokenizers")
TOKENIZERS = {
    m[0]: m[1] for m in inspect.getmembers(sys.modules["tokenizers"], inspect.isclass)
}

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (6144, rlimit[1]))

set_verbosity_error()


os.environ["TOKENIZERS_PARALLELISM"] = "False"


class VisionDataset(BaseDataset):
    _supported = (
        vltk.text,
        vltk.polygons,
        vltk.size,
        vltk.area,
        vltk.boxes,
        vltk.RLE,
    )

    def __init__(
        self,
        config,
        visndatasetadapterdict,
        annotationdict=None,
        metadata_ids=None,
        is_train=False,
        batch_info=None,
        tokenizer_in_visn_dataset=False,
        **kwargs,
    ):

        self.batch_info = batch_info
        self.tokenizer_in_visn_dataset = tokenizer_in_visn_dataset
        if tokenizer_in_visn_dataset:
            self._init_tokenizer(config.lang)
        else:
            self.tokenizer = None
            self.from_transformers = None
        self.is_train = is_train
        # self.annotationdict = annotationdict
        self.config = config
        self.metadata_ids = metadata_ids
        self.annotationdict = annotationdict
        self.metadata_counts = defaultdict(dict)
        for name in annotationdict:
            for meta_name, meta_counts in annotationdict[name]._meta_dict.items():
                self.metadata_counts[name][meta_name] = meta_counts

        self._init_image_processor(config)
        self._init_vision_processors(config)
        self.visndatasetadapterdict = visndatasetadapterdict
        # later if we need
        # print(len(set(annotationdict["funsdaug"].imgids)))

        annotationdict, imgids2pathes, n_imgs = self._shrink_annotation_dicts(
            annotationdict, visndatasetadapterdict
        )
        # print(len(set(annotationdict["funsdaug"].imgids)))

        self._init_annotation_dict(config, annotationdict)
        self.img_id_to_path = imgids2pathes
        self.n_imgs = n_imgs

    def _shrink_annotation_dicts(self, annotationdict, visndatasetadapterdict):
        imgids2pathes = {}
        n_imgs = 0
        for name, imgsetsplits in visndatasetadapterdict.items():
            annodata = annotationdict[name]
            imgids = set()
            for imgids2files in imgsetsplits.values():
                imgids.update(set(imgids2files.keys()))
                imgids2pathes.update(imgids2files)
            filtered_annos = annodata.imgid_filter(imgids, True)
            n_imgs += len(filtered_annos)
            # print("n_imgs", n_imgs)
            annotationdict[name] = filtered_annos
        return annotationdict, imgids2pathes, n_imgs

    @property
    def image(self):
        return self._image

    @property
    def annotations(self):
        return self._annotations

    def _init_vision_processors(self, config):
        vision_processors = (
            config.visn_processors if config.visn_processors is not None else []
        )
        vision_processors = [
            x if not isinstance(x, str) else Processors().get(x)
            for x in vision_processors
        ]

        vision_processors = list(
            filter(lambda x: x.__bases__[0] == VisnProcessor, vision_processors)
        )

        self.vision_processors = [
            x(
                tokenizer=self.tokenizer,
                from_transformers=self.from_transformers,
                config=self.config,
                metadata_ids=self.metadata_ids,
            )
            for x in vision_processors
        ]

        self.vision_processor_keys = ()
        for x in self.vision_processors:
            self.vision_processor_keys += tuple(x.keys)

    def run_vision_processors(self, entry):
        for processor in self.vision_processors:
            entry = processor(entry, config=self.config)
        return entry

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
            try:
                filepath = self.img_id_to_path[img_id]
            except KeyError:
                raise Exception(entry)
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
        entry[vltk.scale] = get_scale(self.image)

        if self.config.ignore_image:
            entry.pop(vltk.img)

        return entry

    @torch.no_grad()
    def _handle_annotations(self, entry, replace_keys=None):
        skip_segmentation = (
            True
            if (vltk.size not in entry or self.config.ignore_segmentation)
            else False
        )
        # get annotations for image
        # entry.update(self.annotations.get(img_id))

        if skip_segmentation and vltk.polygons in entry:
            entry.pop(vltk.polygons)
        if skip_segmentation and vltk.RLE in entry:
            entry.pop(vltk.RLE)

        # run vision processors
        entry = self.run_vision_processors(entry)

        if replace_keys is not None:
            for r in replace_keys:
                if r in entry:
                    entry[vltk.VLOVERLAP[r]] = entry[r]

        return entry

    def __len__(self):
        return self.n_imgs

    @torch.no_grad()
    def __getitem__(self, i):
        anno_dict, anno_dataset = self.annotations[i]
        anno_dict = self._handle_image(anno_dict)
        if self.annotations is not None:
            anno_dict = self._handle_annotations(anno_dict)
        anno_dict = self.try_tensorify(anno_dict)
        self.batch_info.update_entry_keys(anno_dict)
        return anno_dict
