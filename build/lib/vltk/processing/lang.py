import random

import numpy as np
import torch
from vltk.inspection import import_funcs_from_file
import vltk.vars as vltk


class Data:
    def __init__(self):
        if "DATADICT" not in globals():
            global DATADICT
            DATADICT = import_funcs_from_file(vltk.DATAPATH, pkg="vltk.processing")

    def avail(self):
        return list(DATADICT.keys())

    def get(self, name):
        return DATADICT[name]

    def add(self, name, lab):
        DATADICT[name] = lab


def one_hot_label(cur_entry, **kwargs):
    config = kwargs.get("config").lang

    label = cur_entry.get(vltk.label, None)
    score = cur_entry.get(vltk.score, None)
    if label is None:
        label = config.ignore_id
    elif label == config.ignore_id:
        cur_entry.pop(vltk.score, None)
        return
    else:
        if isinstance(label, int):
            pass
        elif len(label) == 1:
            label = label[0]
        else:
            score_sum = sum(score)
            prob = [ss / score_sum for ss in score]
            choice = np.random.multinomial(1, prob).argmax()
            label = label[choice]
    cur_entry.pop(vltk.score, None)
    cur_entry[vltk.label] = label


def multi_hot_label():
    pass


def masked_feature_modeling(cur_entry, **kwargs):
    if "roi_features" not in cur_entry:
        return
    random_feat = kwargs.get("random_feat_func")
    config = kwargs.get("config")
    mask_rate = config.feature_mask_rate
    if config.img_first:
        mask_rate /= 4
    roi_features = cur_entry["roi_features"]
    feat_mask = [0.0] * len(roi_features)
    feat_dim = len(roi_features[0])
    for i in range(len(roi_features)):
        prob = random.random()
        if prob < mask_rate:
            prob /= mask_rate
            # 80% randomly change token to zero feat
            if prob < 0.8:
                roi_features[i] = torch.zeros(feat_dim)

            # 10% randomly change token to random feat
            elif prob < 0.9:
                roi_features[i] = torch.tensor(random_feat())
            # Need to predict this feat
            feat_mask[i] = 1.0
    cur_entry["roi_features"] = roi_features
    cur_entry["feat_mask"] = torch.tensor(feat_mask)


def matched_sentence_modeling(cur_entry, **kwargs):
    random_sents = kwargs.get("random_sents")
    random_sent = lambda: random.choice(random_sents)
    config = kwargs.get("config")
    is_matched = 1
    text = cur_entry[vltk.text]
    rand_text = text
    if random.random() < config.sentence_match_rate:
        if vltk.label in cur_entry:
            cur_entry[vltk.label] = config.ignore_id
            cur_entry[vltk.score] = 0
        is_matched = 0
        while rand_text == text:
            rand_text = random_sent()

    cur_entry["is_matched"] = is_matched
    cur_entry[vltk.text] = rand_text
    return cur_entry


def masked_language_modeling(cur_entry, **kwargs):
    tokenizer = kwargs.get("tokenizer")
    config = kwargs.get("config")
    special_ids = kwargs.get("special_ids")
    n_ids = kwargs.get("n_ids")
    all_ids = kwargs.get("all_ids")
    random_id = lambda: all_ids[random.randint(0, n_ids - 1)]
    # special_ids = set([tokenizer.token_to_id(t) for t in kwargs.get("special_tokens")])
    # random_id = lambda: random.choice(list(tokenizer.get_vocab().items()))[1]

    input_ids = cur_entry["input_ids"]
    attention_mask = cur_entry["text_attention_mask"]
    ignore_id = config.ignore_id
    mask_id = tokenizer.token_to_id("[mask]")
    sep_id = tokenizer.token_to_id("[sep]")
    masked_labels = [ignore_id] * len(input_ids)
    masked_ids = input_ids
    for j, (iid, mid) in enumerate(zip(input_ids[1:], attention_mask[1:]), start=1):
        if int(mid) == 0 or iid == sep_id:
            break
        tid = random_id()
        while tid in special_ids:
            tid = random_id()
        mask_rate = config.word_mask_rate
        prob = random.random()
        if prob < mask_rate:
            old_id = masked_ids[j]
            prob /= mask_rate
            if prob < 0.8:
                masked_ids[j] = mask_id
                pass
            elif prob < 0.9:
                masked_ids[j] = tid
                pass
            masked_labels[j] = old_id
    cur_entry["input_ids"] = masked_ids
    cur_entry["masked_labels"] = masked_labels
