import inspect
# note if we do not immport a pacakage correctly in this class, no loops or exps will be present
import math
import os
import random
import resource
import sys

import torch
import vltk
# disable logging from datasets
from datasets.utils.logging import set_verbosity_error
from vltk.loader.basedataset import (CollatedVLSets, SplitRangesVision,
                                     SplitRangesVL)
from vltk.loader.langdataset import LangDataset
from vltk.loader.visndataset import VisionDataset
from vltk.utils.adapters import Data

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

_data_procecessors = Data()


# TODO
class VisionLanguageDataset(VisionDataset, LangDataset):
    def __init__(
        self,
        config,
        visnlangdatasetadapterdict,  # contains visnlang annotations
        visndatasetadapterdict,  # contains dict of files or dataset of features
        annotationdict=None,  # conatains annotations for vision datasets
        answer_to_id=None,
        object_to_id=None,
        is_train=False,
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
        self.visnlangdatasetadapterdict = visnlangdatasetadapterdict
        self.visndatasetadapterdict = visndatasetadapterdict
        self.vl_idx_organizer = SplitRangesVL(visnlangdatasetadapterdict)
        self.visn_idx_organizer = SplitRangesVision(visndatasetadapterdict)
        """
        TODO:
            I need to make sure all image ids are unique across image ids of various
            datasets. I do not need to worry about this problem now, but later when
            relavent.

            I think the simplest thing to do in the future will be to add an "adjust image
            IDS" function in the vision adapter and not just the vision language adatper
            That way, I can perform a check to make sure both are lined up
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
        self.is_train = is_train
        self.answer_to_id = answer_to_id
        self.object_to_id = object_to_id
        self.uniq_labels = set(answer_to_id.keys())
        splits = self._check_uniq_splits()
        self.splits = splits
        # ======

        # ======
        # do tokenizer stuff
        self._init_tokenizer(config)

        # ======

        # ======
        # prepare image processing components borrowed from visndataset.py
        self._init_annotation_dict(annotationdict)
        self._init_image_processor(config)
        # self.object_to_id = object_to_id
        # self.img_id_to_path = {}
        # self.n_imgs = 0
        # # later if we need
        # self.idx_to_imgid = {}
        # for imgsetsplits in list(visndatasetadapterdict.values()):
        #     for imgids2files in imgsetsplits.values():
        #         self.n_imgs += len(imgids2files)
        #         self.img_id_to_path.update(imgids2files)
        # self.imgids = tuple(self.img_id_to_path.keys())
        # ======

        """
        TODO: figure out how to handle unique ids across datasets that do not refer to the
        same image

        figure out how to handle unique ids across datasets that do refer to the same
        image


        maybe keep track of which image vision dataset it came from?

        idea: prepend dataset id to all images, maybe start prepending all splits aswell

        """

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
                            imgset = visndatasetadapterdict[is_name][is_split]
                            filtered_imgset = imgset.imgid_filter(uniq_imgs, False)
                            # TODO: remove later once I actually end up testing with this
                            assert filtered_imgset.check_imgid_alignment()
                            visndatasetadapterdict[is_name][is_split] = filtered_imgset
                        except Exception:
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
            print(
                f"""
                WARNING: there are no common image IDs between either language or vision datasets,
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
        text_info = self.datasets.get(img_id, return_dataset=True)
        proc_args = self.processor_args()
        text_info = text_info.map(
            lambda x: VisionLanguageDataset.text_map_function(x, proc_args=proc_args)
        )[:]
        for name, item in text_info.items():
            try:
                text_info[name] = torch.tensor(item)
            except Exception:
                pass

        return text_info, img_id

    def _do_map_text_first(self, i):
        entry = self.datasets[i]
        img_id = entry[vltk.imgid]
        proc_args = self.processor_args()
        text_info = self.text_map_function(entry, proc_args)
        for name, item in text_info.items():
            try:
                text_info[name] = torch.tensor(item)
            except Exception:
                pass

        return text_info, img_id

    def random_visn_feat(self):
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
    # TODO: make sure i take into account segmentations because there are a different number of masks per image
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

    def __len__(self):
        if self.config.img_first:
            return int(math.floor(len(self.uniq_imgs) * self.config.percent))
        else:
            return int(math.floor(len(self.datasets) * self.config.percent))

    @torch.no_grad()
    def __getitem__(self, i):
        if self.config.img_first:
            text_info, img_id = self._do_map_img_first(i)
            (
                visnset_name,
                visnset_split,
            ) = self.visn_idx_organizer[i]
            img_info_dict_or_adapter = self.visndatasetadapterdict[visnset_name][
                visnset_split
            ]
            img_info_file_or_entry = img_info_dict_or_adapter.get(img_id)
            if isinstance(img_info_file_or_entry, str):
                img_info_dict = {
                    vltk.filepath: img_info_file_or_entry,
                    vltk.imgid: img_id,
                }
            else:
                raise Exception(
                    img_info_file_or_entry, img_id, visnset_split, visnset_name
                )
            anno_dict = self._handle_image(img_info_dict)
            if self.annotations is not None:
                anno_dict.update(self.annotations.get(img_id))
                self._handle_annotations(anno_dict)
            entry = {**text_info, **anno_dict}
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
            # however, there for all known datasets, there should only be one split present
            for vsplit in vinset_splits:
                try:
                    img_info_dict_or_filepath = self.visndatasetadapterdict[
                        visnset_name
                    ][vsplit].get(img_id)
                except KeyError:
                    pass
            if isinstance(img_info_dict_or_filepath, str):
                img_info_dict = {
                    vltk.filepath: img_info_dict_or_filepath,
                    vltk.imgid: img_id,
                }
            anno_dict = self._handle_image(img_info_dict)
            if self.annotations is not None:
                anno_dict.update(self.annotations.get(img_id))
                self._handle_annotations(anno_dict)
            entry = {**text_info, **anno_dict}
            return entry
