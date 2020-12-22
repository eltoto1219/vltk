import json
import os
from collections import defaultdict
from typing import Tuple, Union

import jsonlines
import numpy as np
import torch
from mllib.utils import get_duration, get_subfiles_from_path

import datasets

PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "libdata")
ANS_CONVERT = json.load(open(os.path.join(PATH, "convert_answers.json")))


def load_arrow(dset_to_arrow_fp: dict, fields: Union[Tuple[str], str, None] = None):
    if fields is not None and not fields:
        return None
    arrow_dict = {}
    for dset in dset_to_arrow_fp:
        arrow_fp = dset_to_arrow_fp[dset]
        arrow = datasets.Dataset.from_file(arrow_fp)
        if fields is not None:
            fields = list(fields)
        arrow.set_format(type="numpy", columns=fields)
        arrow_dict[dset] = arrow
    return arrow_dict


def clip_img_ids(img_ids, percent_data=1.0):
    if percent_data != 1.0:
        stop_int = max(1, int(np.ceil(len(img_ids) * percent_data)))
        img_ids = img_ids[:stop_int]
    assert len(img_ids) > 0
    return img_ids


def process_answer_default(ans):
    if len(ans) == 0:
        return ""
    ans = ans.lower()
    if ans[-1] == ".":
        ans = ans[:-1].strip()
    if ans.startswith("a "):
        ans = ans[2:].strip()
    if ans.startswith("an "):
        ans = ans[3:].strip()
    if ans.startswith("the "):
        ans = ans[4:].strip()
    if ans in ANS_CONVERT:
        ans = ANS_CONVERT[ans]
    return ans


@get_duration
def load_temp_gqa(config, split):
    # get batch_size
    if split in config.train_aliases:
        bz = config.run.batch_size
    else:
        bz = config.evaluate.batch_size
    # labels
    labels = json.load(open(config.pathes.gqa_labels))
    # imgs
    arrow_fp = config.pathes.vg_train_arrow if split in config.train_aliases else config.pathes.vg_test_arrow
    raw_fp = config.pathes.vg_train if split in config.train_aliases else config.pathes.vg_test
    arrow = load_arrow({"gqa": arrow_fp}, config.data.arrow_fields)
    if config.data.use_raw_imgs:
        path_dict = {"gqa": raw_fp}
    else:
        path_dict = None

    # text
    text_fp = config.pathes.gqa_train
    if split in config.valid_aliases:
        text_fp = config.pathes.gqa_valid
    elif split in config.test_aliases:
        text_fp = config.pathes.gqa_test
    elif split in config.eval_aliases:
        text_fp = config.pathes.gqa_testdev

    text_files = get_subfiles_from_path(text_fp)
    print(f"spits to be loaded: {text_files}")
    ignored_labels = set()
    num_ignored = 0
    stop_int = 0
    streams = []
    for text in text_files:
        try:
            data_split = json.load(open(text))
        except json.decoder.JSONDecodeError:
            data_split = jsonlines.open(text)
        stop_int += len(data_split)
        streams.append(data_split)

    if config.dryrun and not config.data.img_first:
        stop_int = config.run.batch_size
    elif not config.dryrun and config.data.percent_data != 1 and not config.data.img_fist:
        stop_int = max(1, int(np.ceil(stop_int * config.data.percent_data)))
    text_data = []
    imgid_to_text = defaultdict(list)
    data_stop = 0
    for stream in streams:
        for data in stream:
            # get base entry
            img_id = str(data["img_id"].split("_")[-1])
            # process label
            label = data["label"]
            assert isinstance(label, dict)
            if len(label) == 0:
                continue
            assert len(label) == 1
            label = next(iter(label.keys()))
            label = process_answer_default(label)
            if label not in labels:
                num_ignored += 1
                ignored_labels.add(label)
                continue
            else:
                entry = {"img_id": img_id, "text": data["sent"], "dset": "gqa", "label": torch.tensor(labels[label])}
                if config.data.img_first:
                    if config.dryrun and len(imgid_to_text) == bz and img_id not in imgid_to_text:
                        pass
                    else:
                        imgid_to_text[img_id].append(entry)
                text_data.append(entry)
            data_stop += 1
            if config.data.img_first and config.dryrun:
                if all(list(map(lambda l: len(l) >= 2, imgid_to_text.values()))):
                    break

    img_ids = list(imgid_to_text.keys())
    if not config.dryrun:
        img_ids = clip_img_ids(img_ids, config.data.percent_data)
    else:
        assert len(img_ids) == bz, (len(img_ids), bz)
        check = list(map(lambda l: len(l) >= 2, imgid_to_text.values()))
        if data_stop != stop_int:
            assert all(check), (imgid_to_text, check)

    assert len(text_data) > 0
    print(f"num ignored: {num_ignored} ignored: {ignored_labels}")
    return {"text": text_data, "pathes": path_dict, "arrow": arrow, "labels": labels, "img_ids": img_ids,
            "imgid_to_text": imgid_to_text}


# def load_temp_lxmert(config, split):

#     coco_valid = datasets.Dataset.from_file(
#         config.pathes.coco_valid_arrow
#     )
#     coco_train = datasets.Dataset.from_file(
#         config.pathes.coco_train_arrow
#     )
#     coco = datasets.concatenate_datasets([coco_valid, coco_train])
#     coco.set_format(type="numpy", output_all_columns=True)
#     vg = datasets.Dataset.from_file(config.pathes.vg_arrow)
#     vg.set_format(type="numpy", output_all_columns=True)
#     arrow_dict = {"coco": coco, "vg": vg}

#     path_dict = {
#         "coco": config.pathes.coco_imgs,
#         "vg": config.pathes.vg_imgs,
#     }

#     if split in ("pretrain", "train", "finetune"):
#         text_files = get_subfiles_from_path(
#             config.pathes.temp_lxmert_train
#         )
#     elif split in ("eval", "evaluation", "validation", "val"):
#         text_files = get_subfiles_from_path(
#             config.pathes.temp_lxmert_eval)
#     else:
#         text_files = get_subfiles_from_path(
#             config.pathes.temp_lxmert_test
#         )

#     labels = set(
#         [
#             process_answer_default(ans["ans"])
#             for ans in json.load(
#                 open(config.pathes.temp_lxmert_answers)
#             )
#         ]
#     )

#     name2dset = {
#         "mscoco": "coco",
#         "vg": "vg",
#         "visual7w": "vg",
#         "gqa": "vg",
#         "vqa": "coco",
#     }
#     print("loading text data")
#     num_labels = 0
#     label2lid = {}
#     ignore_idx = -100
#     text_data = []
#     for text in text_files:
#         data_split = json.load(open(text))
#         for data in data_split:
#             img_id = data["img_id"].split("_")[-1]
#             sentf = data["sentf"]
#             for sents_cat, sents in sentf.items():
#                 dset = name2dset[sents_cat]
#                 if sents_cat in data["labelf"]:
#                     labels = data["labelf"][sents_cat]
#                 else:
#                     labels = None
#                 for sent_idx, sent in enumerate(sents):
#                     entry = {"img_id": img_id, "text": sent, "dset": dset}
#                     if labels is not None:
#                         label = labels[sent_idx]
#                         in_label_set = False
#                         if not len(label) == 0:
#                             label = OrderedDict(
#                                 {
#                                     k: v
#                                     for k, v in sorted(
#                                         label.items(),
#                                         key=lambda item: item[1],
#                                         reverse=True,
#                                     )
#                                 }
#                             )
#                             for k in label.keys():
#                                 k = process_answer_default(k)
#                                 if k in labels:
#                                     in_label_set = True
#                                     label = k
#                                     break
#                             if in_label_set:
#                                 if label not in labels:
#                                     lid = num_labels
#                                     label2lid[label] = lid
#                                     num_labels += 1
#                                 else:
#                                     lid = label2lid[label]
#                             else:
#                                 lid = ignore_idx
#                     else:
#                         lid = ignore_idx
#                     entry["label"] = torch.tensor(lid)
#                     text_data.append(entry)

#     return text_data, path_dict, arrow_dict, labels


NAME2DATASET = {"temp_lxmert": None, "gqa": load_temp_gqa}

"""
    def matched_sentence_modeling(self, entry):
        is_matched = 1
        if random.random() < self.dataset_config.sentence_match_rate:
            is_matched = 0
            other_datum = self.text_data[random.randint(0, len(self.text_data) - 1)]
            while other_datum["img_id"] == entry["img_id"]:
                other_datum = self.text_data[random.randint(0, len(self.text_data) - 1)]
            sent = other_datum["text"]
            entry["text"] = sent
        if not is_matched:
            entry["label"] = torch.tensor(self.ignore_id)

        is_matched = torch.tensor(is_matched)
        entry["is_matched"] = is_matched
        return entry

    def masked_language_modeling(self, input_ids):
        ignore_id = self.ignore_id
        mask_id = self.mask_id
        masked_sequence = input_ids
        masked_inds = torch.zeros(input_ids.shape)
        # do random_sampling instead of iteration: faster
        for i in range(len(masked_sequence)):
            ind = int(masked_sequence[i])
            random_id = self.random_ind()
            while random_id in self.special_ids:
                random_id = self.random_ind()
            prob = random.random()
            ratio = self.dataset_config.word_mask_rate
            if prob < ratio and ind not in self.special_ids:
                prob /= ratio
                if prob < 0.8:
                    masked_sequence[i] = mask_id
                elif prob < 0.9:
                    masked_sequence[i] = random_id
                assert ind not in self.special_ids
                masked_inds[i] = ind
            else:
                masked_inds[i] = ignore_id

        return masked_inds, masked_sequence

    def get_random_ind(self):
        token = random.choice(list(self.tokenizer.vocab.items()))
        return int(token[-1])

"""

