import random
import torch
from copy import deepcopy

def one_hot_label():
    pass

def multi_hot_label():
    pass

    # if example.label is None or len(example.label) == 0 or example.is_matched != 1:
    #     # 1. No label 2. Label is pruned 3. unmatched visual + language pair
    #     ans = -1
    # else:
    #     keys, values = zip(*example.label.items())
    #     if len(keys) == 1:
    #         ans = keys[0]
    #     else:
    #         value_sum = sum(values)
    #         prob = [value / value_sum for value in values]
    #         choice = np.random.multinomial(1, prob).argmax()
    #         ans = keys[choice]

def masked_feature_modeling(dataset_object, cur_entry):
     random_feat(feats):
    mask_feats = feats.copy()
    feat_mask = np.zeros(len(feats), dtype=np.float32)
    # for i in range(len(feats)):
    #     prob = random.random()
    #     # mask token with probability
    #     if prob < args.obj_mask_rate:
    #         prob /= args.obj_mask_rate

    #         # 80% randomly change token to zero feat
    #         if prob < 0.8:
    #             mask_feats[i, :] = 0.

    #         # 10% randomly change token to random feat
    #         elif prob < 0.9:
    #             mask_feats[i, :] = train_tuple.torchdset.random_feat()
    #         # -> rest 10% randomly keep current feat

    #         # Need to predict this feat
    #         feat_mask[i] = 1.

    return mask_feats, feat_mask

def matched_sentence_modeling(dataset_object, cur_entry):
    mask_rate = dataset_object.config.sentence_match_rate:
    is_matched = 1
    if random.random() <  mask_rate:
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

def masked_language_modeling(dataset_object, cur_entry):
    input_ids = cur_entry["input_ids"]
    ignore_id = dataset_object.config.ignore_id
    mask_token = "[mask]"
    mask_id = dataset_object.tokenizer.token_to_id(mask_token)
    take_first = False
    if input_ids.dim() == 1:
        take_first = True
        input_ids = [input_ids]
        attention_mask = [input_ids]
    masked_seqs = []
    masked_id_seqs = []
    for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
        masked_inds = [0] * len(ids)
        masked_seq = deepcopy(ids.to_list())
        for id_idx, mask_idx in zip(ids, mask):
            if int(mask_idx) == 0:
                break
            while token in dataset_object.special_tokens:
                (token, tid) = random.choice(list(dataset_object.tokenizer.get_vocab().items()))
            mask_rate = dataset_object.config.word_mask_rate
            prob = random.random()
            if prob < mask_rate:
                prob /= mask_rate
                if prob < 0.8:
                    masked_seq[i] = mask_id
                    pass
                elif prob < 0.9:
                    masked_seq[i] = random_id
                    pass
                masked_inds[i] = masked_seq[i]k
            else:
                masked_inds[i] = ignore_id
        masked_id_seqs.append(torch.tensor(masked_ids))
        masked_seqs.append(torch.tensor(masked_seq))
    entry["masked_ids"] = masked_seqs
    entry["masked_labels"] =  masked_id_seqs

