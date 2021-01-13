import random
import torch
from vltk import LABELKEY, SCOREKEY
from copy import deepcopy
import numpy as np

def one_hot_label(dataset_object, cur_entry):
    label = cur_entry.get(LABELKEY, None)
    score = cur_entry.get(SCOREKEY, None)
    take_first = False
    if not isinstance(label, list):
        take_first = True
        label = [label]
        score = [score]

    new_labels = [0] * len(label)
    for i, (l, s) in enumerate(zip(label, score)):
        if l is None:
            new_labels[i] = dataset_object.config.ignore_id
        else:
            if len(l) == 1:
                new_labels[i] = l
            else:
                score_sum = sum(s)
                prob = [ss / score_sum for ss in s]
                choice = np.random.multinomial(1, prob).argmax()
                new_labels[i] = l[choice]
    cur_entry.pop(SCOREKEY, None)
    cur_entry[LABELKEY] = torch.Tensor(new_labels) if not take_first else new_labels[0]


def multi_hot_label():
    pass


def masked_feature_modeling(dataset_object, cur_entry):
    if "roi_features" not in cur_entry:
        return
    mask_rate = dataset_object.config.feature_mask_rate
    if dataset_object.config.img_first:
        mask_rate /= 4
    roi_features = cur_entry["roi_features"]
    feat_mask = np.zeros(len(roi_features), dtype=np.float32)
    for i in range(len(roi_features)):
        prob = random.random()
        if prob < mask_rate:
            prob /= args.obj_mask_rate
            # 80% randomly change token to zero feat
            if prob < 0.8:
                roi_features[i, :] = 0.

            # 10% randomly change token to random feat
            elif prob < 0.9:
                roi_features[i, :] = dataset_object.random_feat()
            # Need to predict this feat
            feat_mask[i] = 1.
    cur_entry["roi_features"] = roi_features
    cur_entry["feat_mask"] = torch.tensor(feat_mask)


def matched_sentence_modeling(dataset_object, cur_entry):
    mask_rate = dataset_object.config.sentence_match_rate
    input_ids = cur_entry["input_ids"]
    attention_mask = cur_entry["text_attention_mask"]
    type_ids = cur_entry["type_ids"]
    take_first = False
    if not isinstance(input_ids, list):
        take_first = True
        input_ids = [input_ids]
        attention_mask = [attention_mask]
        type_ids = [type_ids]
    matched = [1] * len(input_ids)
    for i, (seq, mask, id_t) in enumerate(zip(input_ids, attention_mask, type_ids)):
        is_matched = 1
        if random.random() <  mask_rate:
            is_matched = 0
            o_ids = seq.tolist()
            while torch.all(seq.eq(torch.Tensor(o_ids))):
                other_dict = dataset_object.random_sent()
                o_ids = other_dict["input_ids"]
                o_tids  = other_dict["type_ids"]
                o_mask = other_dict["text_attention_mask"]
            input_ids[i] = o_ids
            type_ids[i] = o_tids
            attention_mask[i] = o_mask
        if not is_matched:
            input_ids[i]  = dataset_object.config.ignore_id
            matched[i] = 0

    matched = torch.tensor(matched)
    cur_entry["is_matched"] = matched if not take_first else matched[0]
    cur_entry["input_ids"] = input_ids if not take_first else input_ids[0]
    cur_entry["text_attention_mask"] = attention_mask if not take_first else attention_mask[0]
    cur_entry["type_ids"] = type_ids if not take_first else type_ids[0]

def masked_language_modeling(dataset_object, cur_entry):
    input_ids = cur_entry["input_ids"]
    attention_mask= cur_entry["text_attention_mask"]
    ignore_id = dataset_object.config.ignore_id
    mask_token = "[mask]"
    mask_id = dataset_object.tokenizer.token_to_id(mask_token)
    take_first = False
    if input_ids.dim() == 1:
        take_first = True
        input_ids = [input_ids]
        attention_mask = [attention_mask]
    masked_seqs = []
    masked_id_seqs = []
    for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
        special_ids = set([dataset_object.tokenizer.token_to_id(j) for j in dataset_object.special_tokens])
        masked_inds = [ignore_id] * len(ids)
        masked_seq = deepcopy(ids.tolist())
        for j, (id_idx, mask_idx) in enumerate(zip(ids[1:], mask[1:]), start=1):
            if int(mask_idx) == 0:
                break
            tid = dataset_object.random_id()
            while tid in special_ids:
                tid = dataset_object.random_id()
            mask_rate = dataset_object.config.word_mask_rate
            prob = random.random()
            if prob < mask_rate:
                old_id = masked_seq[j]
                prob /= mask_rate
                if prob < 0.8:
                    masked_seq[j] = mask_id
                    pass
                elif prob < 0.9:
                    masked_seq[j] = tid
                    pass
                masked_inds[j] = old_id
        masked_id_seqs.append(torch.tensor(masked_inds))
        masked_seqs.append(torch.tensor(masked_seq))
    cur_entry["input_ids"] = masked_seqs if not take_first else masked_seqs[0]
    cur_entry["masked_labels"] =  masked_id_seqs if not take_first else masked_id_seqs[0]
