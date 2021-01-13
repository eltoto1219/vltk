

def matched_sentence_modeling(dataset_object, cur_entry):
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

def masked_language_modeling(dataset_object, cur_entry):
    input_ids = cur_entry["input_ids"]
    raise Exception(input_ids)
    raise Exception("woot")
    # need to print shape before contining
    # attention_mask = cur_entry["input_ids"]
    # masked_inds =

    # for i in range(len(masked_sequence)):
    #     ind = int(masked_sequence[i])
    #     random_id = self.random_ind()
    #     while random_id in self.special_ids:
    #         random_id = self.random_ind()
    #     prob = random.random()
    #     ratio = self.dataset_config.word_mask_rate
    #     if prob < ratio and ind not in self.special_ids:
    #         prob /= ratio
    #         if prob < 0.8:
    #             masked_sequence[i] = mask_id
    #         elif prob < 0.9:
    #             masked_sequence[i] = random_id
    #         assert ind not in self.special_ids
    #         masked_inds[i] = ind
    #     else:
    #         masked_inds[i] = ignore_id

    # return masked_inds, masked_sequence

