import inspect
import math
import os
import random
import resource
import sys

import torch
import vltk.vars as vltk
# disable logging from datasets
from datasets.utils.logging import set_verbosity_error
from vltk.dataset.basedataset import (CollatedVLSets, SplitRangesVision,
                                      SplitRangesVL)
from vltk.dataset.langdataset import LangDataset
from vltk.dataset.visndataset import VisionDataset
from vltk.processing import Processors, VisnLangProcessor

__import__("tokenizers")
TOKENIZERS = {
    m[0]: m[1] for m in inspect.getmembers(sys.modules["tokenizers"], inspect.isclass)
}

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (6144, rlimit[1]))

set_verbosity_error()


os.environ["TOKENIZERS_PARALLELISM"] = "False"


# TODO
class VisionLanguageDataset(VisionDataset, LangDataset):
    visn = set()
    lang = set()

    def __init__(
        self,
        config,
        visnlangdatasetadapterdict,  # contains visnlang annotations
        visndatasetadapterdict,  # contains dict of files or dataset of features
        annotationdict=None,  # conatains annotations for vision datasets
        metadata_ids=None,
        is_train=False,
        batch_info=None,
        tokenizer_in_visn_dataset=False,
        replace_keys=None,
        **kwargs,
    ):
        # ======
        # checking/setting respective adatpers
        uniq_imgs, missing_ids, shrink_lang, shrink_vision = self._check_uniq_imgs(
            visndatasetadapterdict, visnlangdatasetadapterdict
        )
        visndatasetadapterdict, visnlangdatasetadapterdict = self._tighten_datasets(
            uniq_imgs,
            visndatasetadapterdict,
            visnlangdatasetadapterdict,
            missing_ids,
            shrink_lang,
            shrink_vision,
        )
        self.tokenizer_in_visn_dataset = tokenizer_in_visn_dataset

        self.visnlangdatasetadapterdict = visnlangdatasetadapterdict
        self.visndatasetadapterdict = visndatasetadapterdict
        self.vl_idx_organizer = SplitRangesVL(visnlangdatasetadapterdict)
        self.visn_idx_organizer = SplitRangesVision(visndatasetadapterdict)
        # raise Exception(
        #     self.visn_idx_organizer.uniq_imgs, self.vl_idx_organizer.uniq_imgs
        # )
        """
        WARNING:
            untested: ensure all image ids are unique across image ids of various
            datasets.

            the simplest thing to do in the future will be to add an "adjust image
            IDS" function in the vision adapter and not just the vision language adatper
            That way, a check can be performed to make sure both are lined up
        """
        self.uniq_imgs = self.visn_idx_organizer.imgs
        visnlangdatasetadapters = []
        for dset in self.visnlangdatasetadapterdict:
            for split in self.visnlangdatasetadapterdict[dset]:
                visnlangdatasetadapters.append(
                    self.visnlangdatasetadapterdict[dset][split]
                )
        self.datasets = CollatedVLSets(*visnlangdatasetadapters)
        self.annotationdict = annotationdict
        # ======

        # ======
        # set some other properties
        self.config = config
        self.replace_keys = replace_keys
        self.batch_info = batch_info
        self.is_train = is_train
        self.metadata_ids = metadata_ids
        splits = self._check_uniq_splits()
        self.splits = splits
        self.placeholders = {}
        self.max_spanning_cols = kwargs.get("max_spanning_cols")
        # ======

        # ======
        # do tokenizer stuff
        self._init_tokenizer(config.lang)
        self._init_annotation_dict(config, annotationdict)
        self._init_image_processor(config)
        self._init_vision_processors(config)
        self._init_lang_processors(config)
        self._init_visnlang_processors(config)

        # ======

    def update_visn_lang_keys(self, lang_entry, visn_entry):
        self.visn = self.visn.union(set(visn_entry.keys()))
        self.lang = self.lang.union(set(lang_entry.keys()))

    def _init_visnlang_processors(self, config):
        visnlang_processors = (
            config.visnlang_processors if config.visnlang_processors is not None else []
        )

        visnlang_processors = [
            x if not isinstance(x, str) else Processors().get(x)
            for x in visnlang_processors
        ]

        visnlang_processors = list(
            filter(lambda x: x.__bases__[0] == VisnLangProcessor, visnlang_processors)
        )

        self.visnlang_processors = [
            x(
                config=self.config,
                metadata_ids=self.metadata_ids,
                from_transformers=self.from_transformers,
                tokenizer=self.tokenizer,
            )
            for x in visnlang_processors
        ]

        self.visnlang_processor_keys = ()
        for x in self.visnlang_processor_keys:
            self.visnlang_processor_keys += tuple(x.keys)

    def run_visnlang_processors(self, lang_entry, visn_entry, img_first):
        for processor in self.visnlang_processors:
            lang_entry, visn_entry = processor(
                lang_entry, visn_entry, img_first=img_first
            )
        return lang_entry, visn_entry

    def _tighten_datasets(
        self,
        uniq_imgs,
        visndatasetadapterdict,
        visnlangdatasetadapterdict,
        missing_ids,
        shrink_lang,
        shrink_vision,
    ):
        if missing_ids is not None:
            print(f"resizing datasets to account for {missing_ids} missing image IDs")
            if shrink_lang:
                for dset in visnlangdatasetadapterdict:
                    for split in visnlangdatasetadapterdict[dset]:
                        visnlang = visnlangdatasetadapterdict[dset][split]
                        filtered_visnlang = visnlang.imgid_filter(uniq_imgs, True)
                        visnlangdatasetadapterdict[dset][split] = filtered_visnlang

            if shrink_vision:
                for is_name in visndatasetadapterdict:
                    for is_split in visndatasetadapterdict[is_name]:
                        try:
                            # for visionsets that are Dataset objects
                            imgset = visndatasetadapterdict[is_name][is_split]
                            filtered_imgset = imgset.imgid_filter(uniq_imgs, False)
                            # TODO: remove later once I actually end up testing with this
                            # assert filtered_imgset.check_imgid_alignment()
                            visndatasetadapterdict[is_name][is_split] = filtered_imgset
                        except Exception:
                            # for vision sets that are only a dictionary of imgids
                            imgsetdict = visndatasetadapterdict[is_name][is_split]
                            imgsetdict = dict(
                                filter(lambda x: x[0] in uniq_imgs, imgsetdict.items())
                            )

                            visndatasetadapterdict[is_name][is_split] = imgsetdict

        return visndatasetadapterdict, visnlangdatasetadapterdict

    def _check_uniq_imgs(self, visndatasetadapterdict, visnlangdatasetadapterdict):
        uniq_visn_imgs = set()
        for is_info in visndatasetadapterdict.items():
            is_name, is_split_dict = is_info
            for k, imgset in is_split_dict.items():
                try:
                    img_ids = imgset.imgids
                    # Adapters().get(imgset.dataset).adjust_imgid()
                    # raise Exception(k, imgset, imgset.dataset)
                except Exception:
                    img_ids = imgset.keys()

                img_ids = set(img_ids)

                uniq_visn_imgs = uniq_visn_imgs.union(img_ids)
        uniq_lang_imgs = set()
        uniq_imgs = set()
        for ts_name, ts_splits in visnlangdatasetadapterdict.items():
            for split_name, ts in visnlangdatasetadapterdict[ts_name].items():
                temp_uniq = ts.uniq_imgs
                uniq_lang_imgs = uniq_lang_imgs.union(uniq_lang_imgs, temp_uniq)
                uniq_imgs = uniq_imgs.union(uniq_visn_imgs.intersection(temp_uniq))
        if not uniq_imgs:
            raise Exception(
                f"""
                ERROR: there are no common image IDs between either language or vision datasets,
                you may want to rename them or check to see if this should be the case or implement
                the `adjust_imgid` function in the VisnLangAdapter. \n
                Vision Dataset Image ID example: {next(iter(uniq_visn_imgs))}
                Language Dataset Image ID example: {next(iter(uniq_lang_imgs))}
                """
            )
        missing_ids = None
        shrink_lang = False
        shrink_vision = False
        all_imgs = uniq_lang_imgs.union(uniq_visn_imgs)
        if len(uniq_imgs) < len(all_imgs):
            missing_ids = len(all_imgs) - len(uniq_imgs)
            if not len(uniq_imgs) == len(uniq_visn_imgs):
                shrink_vision = True
            if not len(uniq_imgs) == len(uniq_lang_imgs):
                shrink_lang = True
        return uniq_imgs, missing_ids, shrink_lang, shrink_vision

    def _check_uniq_splits(self):
        splits = set()
        for v in self.visnlangdatasetadapterdict.values():
            splits = splits.union(set(v.keys()))
        return splits

    def _do_map_img_first(self, i):
        img_id = self.uniq_imgs[i]
        text_info = self.datasets.get(img_id)
        # try:
        # except Exception:
        #     for a in self.datasets.args:
        #         print(set(a["imgid"]))
        #     print("---")
        #     text_info = self.datasets.get(str(img_id))
        #     raise Exception(self.uniq_imgs, str(img_id))
        text_info.pop(vltk.imgid)
        text_info = self._handle_text_annotations(text_info, encode_batch=True)
        return text_info, img_id

    def _do_map_text_first(self, i):
        text_info = self.datasets[i]
        img_id = text_info[vltk.imgid]
        text_info = self._handle_text_annotations(text_info, encode_batch=False)
        return text_info, img_id

    # def random_visn_feat(self):
    #     rand_ind = random.randint(0, len(self.uniq_imgs) - 1)
    #     img_id = self.uniq_imgs[rand_ind]
    #     ts_name, ts_split = self.img2visnlangdatasetadapter[img_id]
    #     visnlangdatasetadapter = self.visnlangdatasetadapterdict[ts_name][ts_split]
    #     is_name, is_split = zip(*visnlangdatasetadapter.data_info[ts_split].items())
    #     visndatasetadapter = self.visndatasetadapterdict[is_name[0]][is_split[0][0]]
    #     img_info = visndatasetadapter.get(img_id)
    #     if vltk.features in img_info:
    #         feat = random.choice(img_info[vltk.features])
    #         return feat
    #     else:
    #         return None

    def transpose_vl(self, batch, max_size=512):
        visn_keys = tuple(self.visn)
        lang_keys = tuple(self.lang)
        if not visn_keys or not lang_keys:
            raise Exception(
                f"User must iterate through the  loader atleast once to use this function: {self.visn & self.lang},\
                {self.test}"
            )

        # first we resize image according to how many examples that we need
        val = None
        for x in lang_keys:
            val = batch.get(x, None)
            if val is not None:
                break
        n_exs_per_img = [len(i) for i in val]

        # process visn keys first
        device = None
        for visn_key in visn_keys:
            v = batch.get(visn_key, None)
            if v is None:
                continue
            if not isinstance(v, list):
                visn_val = torch.cat(
                    [
                        i.unsqueeze(0).expand(min(n, max_size), *i.shape)
                        for i, n in zip(v, n_exs_per_img)
                    ],
                    dim=0,
                )
                batch[visn_key] = visn_val
            else:
                visn_val = [[v] * min(n, max_size) for i, n in zip(v, n_exs_per_img)]
                batch[visn_key] = visn_val
            if device is not None:
                batch[visn_key] = batch[visn_key].to(device)

        # now we flatten the nested lang keys
        for lang_key in lang_keys:
            v = batch.get(lang_key, None)
            if v is None:
                continue
            if isinstance(v, list):
                if isinstance(v[0], torch.Tensor):
                    lang_val = torch.cat(
                        [j[: min(max_size, n)] for j, n in zip(v, n_exs_per_img)],
                        dim=0,
                    )
                    batch[lang_key] = lang_val
                elif isinstance(v[0], str):
                    lang_val = []
                    # here is also a part that we want to recude
                    for i, n in zip(v, n_exs_per_img):
                        if n >= max_size:
                            n = min(n, max_size)
                        lang_val.extend(i * n)
                    batch[lang_key] = lang_val
            else:
                lang_val = torch.stack(
                    [j[: min(max_size, n)] for j, n in zip(v, n_exs_per_img)],
                    dim=0,
                )
                batch[lang_key] = lang_val

        # raise Exception(batch)
        return batch

    def __len__(self):
        if self.config.img_first:
            return int(math.floor(len(self.uniq_imgs) * self.config.percent))
        else:
            return int(math.floor(len(self.datasets) * self.config.percent))

    @torch.no_grad()
    def __getitem__(self, i):
        if self.config.img_first:
            text_info, img_id = self._do_map_img_first(i)
            # raise Exception(len(self), img_id, i, self.uniq_imgs)
            (
                visnset_name,
                visnset_split,
            ) = self.visn_idx_organizer[i]
            img_info_dict_or_adapter = self.visndatasetadapterdict[visnset_name][
                visnset_split
            ]
            anno_dict_or_str = img_info_dict_or_adapter.get(img_id)
            if isinstance(anno_dict_or_str, str):
                anno_dict = {
                    vltk.filepath: anno_dict_or_str,
                    vltk.imgid: img_id,
                }
                anno_dict = self._handle_image(anno_dict)
            else:
                anno_dict = anno_dict_or_str

            # update with ground truth annotations, allow predcitions to overwrite GT
            if self.annotations is not None:
                annotations = self.annotations.get(img_id)
                annotations = {
                    k: v for k, v in annotations.items() if k not in anno_dict
                }
                anno_dict.update(annotations)

            anno_dict = self._handle_annotations(
                anno_dict,
                replace_keys=self.replace_keys,
            )
            text_info, anno_dict = self.run_visnlang_processors(
                text_info, anno_dict, self.config.img_first
            )

            entry = {**text_info, **anno_dict}
            # self.batch_info.update_visn_lang_keys(text_info, anno_dict)
            self.update_visn_lang_keys(text_info, anno_dict)
            entry = self.try_tensorify(entry)
            self.batch_info.update_entry_keys(entry)

            return entry

        else:
            # lets allow this to be the default
            text_info, img_id = self._do_map_text_first(i)
            (
                lang_split,
                langset_name,
                visnset_name,
                vinset_splits,
            ) = self.vl_idx_organizer[i]
            # TODO: for now I just check through all splits to see which imgid is present.
            # however, for all known datasets, there should only be one split present
            img_info_dict_or_filepath = None
            for vsplit in vinset_splits:
                visnadapter = self.visndatasetadapterdict[visnset_name][vsplit]
                try:
                    img_info_dict_or_filepath = visnadapter.get(img_id, None)
                except KeyError:
                    img_info_dict_or_filepath = None
                    continue

                if img_info_dict_or_filepath is None:
                    pass
                elif not isinstance(img_info_dict_or_filepath, (dict, str)):
                    img_info_dict_or_filepath = img_info_dict_or_filepath[0]
                    if not isinstance(img_info_dict_or_filepath, (dict, str)):
                        raise Exception(img_info_dict_or_filepath)

            if isinstance(img_info_dict_or_filepath, str):
                anno_dict = {
                    vltk.filepath: img_info_dict_or_filepath,
                    vltk.imgid: img_id,
                }
                anno_dict = self._handle_image(anno_dict)
            else:
                anno_dict = img_info_dict_or_filepath

            if self.annotations is not None:
                annotations = self.annotations.get(img_id)
                annotations = {
                    k: v for k, v in annotations.items() if k not in anno_dict
                }
                anno_dict.update(annotations)

            self._handle_annotations(
                anno_dict,
                replace_keys=self.replace_keys,
            )

            text_info, anno_dict = self.run_visnlang_processors(
                text_info, anno_dict, self.config.img_first
            )

            entry = {**text_info, **anno_dict}
            # self.batch_info.update_visn_lang_keys(text_info, anno_dict)
            self.update_visn_lang_keys(text_info, anno_dict)
            entry = self.try_tensorify(entry)
            self.batch_info.update_entry_keys(entry)
            return entry
