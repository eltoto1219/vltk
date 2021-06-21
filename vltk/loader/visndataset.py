import inspect
# note if we do not immport a pacakage correctly in this class, no loops or exps will be present
import json
import os
import resource
import sys
from itertools import chain

import torch
import vltk
from datasets.utils.logging import set_verbosity_error
# disable logging from datasets
from vltk.loader.basedataset import BaseDataset, CollatedVisionSets
from vltk.processing import Processors, VisnProccessor
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

VOCABPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "libdata/bert-base-uncased-vocab.txt")
).replace("loader/", "")
TOKENIZEDKEY = "encoded"
os.environ["TOKENIZERS_PARALLELISM"] = "False"


class VisionDataset(BaseDataset):
    _supported = (
        vltk.text,
        vltk.polygons,
        vltk.size,
        vltk.area,
        vltk.box,
        vltk.RLE,
        vltk.tokenbox,
    )
    _padding = {vltk.box: [0, 0, 0, 0]}
    _sep = {vltk.box: torch.tensor([[0, 0, 0, 0]])}
    _cls = {vltk.box: torch.tensor([[0, 0, 0, 0]])}

    def __init__(
        self,
        config,
        visndatasetadapterdict,
        annotationdict=None,
        object_to_id=None,
        is_train=False,
        all_same_keys=True,
        tokenizer_in_visn_dataset=False,
        **kwargs,
    ):

        self.tokenizer_in_visn_dataset = tokenizer_in_visn_dataset
        if tokenizer_in_visn_dataset:
            self._init_tokenizer(config)
        self.is_train = is_train
        # self.annotationdict = annotationdict
        self._init_annotation_dict(config, annotationdict)
        self.config = config
        self._init_image_processor(config)
        self._init_vision_processors(config)
        self._init_box_cls_token(config)
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
        self.max_spanning_cols = kwargs.get("max_spanning_cols", None)

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

    def _init_box_cls_token(self, config):
        size = self.config.visn.size
        if isinstance(size, tuple):
            cls_box = [size[0], size[1], 0, 0]
        else:
            cls_box = [size, size, 0, 0]
        self._cls_box = cls_box

    @torch.no_grad()
    def cls_box(self):
        return torch.tensor([self._cls_box])

    def _init_vision_processors(self, config):
        vision_processors = config.processors if config.processors is not None else []
        vision_processors = [
            x if not isinstance(x, str) else Processors().get(x)
            for x in vision_processors
        ]

        vision_processors = list(
            filter(lambda x: x.__bases__[0] == VisnProccessor, vision_processors)
        )

        self.vision_processors = [x() for x in vision_processors]

        self.vision_processor_keys = ()
        for x in self.vision_processors:
            self.vision_processor_keys += x.keys

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

    @torch.no_grad()
    def _handle_annotations(self, entry, replace_keys=None, extra_features=None):
        img_id = entry[vltk.imgid]
        skip_segmentation = (
            True
            if (vltk.size not in entry or self.config.ignore_segmentation)
            or not self.config.add_cls_to_box
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
        if vltk.label in entry and not isinstance(entry[vltk.label], torch.Tensor):
            word_labels = entry[vltk.label]
            labels = torch.Tensor(self.answer_to_id[word_labels])
            entry[vltk.label] = labels
        if vltk.objects in entry and not isinstance(entry[vltk.objects], torch.Tensor):
            word_labels = entry[vltk.objects]
            n_objects = len(word_labels)
            entry[vltk.n_objects] = torch.tensor(n_objects)
            if n_objects == self.config.max_objects and self.config.add_cls_to_box:
                word_labels[-1] = ""
            word_labels += [""] * max(0, (self.config.max_objects - n_objects))
            labels = torch.Tensor([self.object_to_id[l] for l in word_labels])
            entry[vltk.objects] = labels

        # go through annotations not in processors or supported
        for k, v in entry.items():
            if (
                k not in (vltk.objects, vltk.n_objects)
                and not isinstance(v, torch.Tensor)
                and k not in self.vision_processor_keys
                and k not in vltk.SUPPORTEDNAMES
            ):
                # take care of lists of strings
                try:
                    prim = base.get_list_primitive(v)
                except Exception:
                    raise Exception(k, v)
                if prim == str:
                    values = base.convertids_recursive(v, self.object_to_id)
                    entry[k] = values

        # run vision processors
        entry = self.run_vision_processors(entry)

        # only loop through annotations processed by vltk
        for k in self._supported:
            if k not in entry or isinstance(entry[k], torch.Tensor):
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

            elif k == vltk.box or k == vltk.tokenbox:
                values = v
                if k == vltk.box:
                    if self.config.add_cls_to_box:
                        values = values[: min(len(values), self.config.max_objects)]
                    else:
                        values = values[: min(len(values), self.config.max_objects - 1)]
                else:
                    values = values[
                        : min(len(values), self.config.lang.max_visual_seq_length - 1)
                    ]
                values = torch.tensor(v)

                if vltk.scale in entry:
                    values = rescale_box(values, entry[vltk.scale])
                if k == vltk.tokenbox:
                    values = torch.nn.functional.pad(
                        values,
                        (
                            0,
                            0,
                            0,
                            (self.config.lang.max_visual_seq_length - 1) - len(values),
                        ),
                    )
                    values = torch.cat((values, self._sep[vltk.box]), dim=0)
                else:
                    if self.config.add_cls_to_box:
                        values = torch.cat((values, self.cls_box()), dim=0)

                    values = torch.nn.functional.pad(
                        values,
                        (0, 0, 0, self.config.max_objects - len(values)),
                    )
                entry[k] = values
            elif k == vltk.text:
                if not self.from_transformers:
                    self.disable_padding()
                    text = list(
                        map(
                            lambda x: x.ids,
                            self.tokenizer.encode_batch(
                                entry.pop(k), add_special_tokens=False
                            ),
                        )
                    )
                    self.enable_padding()
                else:
                    text = self.tokenizer(
                        entry.pop(k),
                        add_special_tokens=False,
                        return_attention_mask=False,
                    )["input_ids"]
                # I will always ensure tokenbox comes after entry
                if extra_features is not None and vltk.span in extra_features:
                    spans = [
                        list(chain(*map(lambda x: [x[0]] * len(x[1]), zip(span, text))))
                        for span in extra_features.pop(vltk.span)
                    ]
                    spans = torch.tensor(
                        list(
                            map(
                                lambda x: x[
                                    : min(
                                        self.config.lang.max_visual_seq_length, len(x)
                                    )
                                ]
                                + [0]
                                * (
                                    self.config.lang.max_visual_seq_length
                                    - min(
                                        self.config.lang.max_visual_seq_length, len(x)
                                    )
                                ),
                                spans,
                            )
                        )
                    )
                    entry[vltk.span] = spans

                if vltk.tokenbox in entry:
                    entry[vltk.tokenbox] = list(
                        chain(
                            *map(
                                lambda x: [x[0]] * len(x[1]),
                                zip(entry.get(vltk.tokenbox), text),
                            )
                        )
                    )

                text = list(chain(*text))[
                    : min(len(text), self.config.lang.max_visual_seq_length - 1)
                ]
                if not self.from_transformers:
                    text += [self.tokenizer.token_to_id(self.tokenizer.sep_token)]
                else:
                    text += [
                        self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
                    ]
                text += [0] * (self.config.lang.max_visual_seq_length - len(text))
                entry[k] = torch.tensor(text)

        if replace_keys is not None:
            for r in replace_keys:
                if r in entry:
                    entry[vltk.VLOVERLAP[r]] = entry[r]

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
