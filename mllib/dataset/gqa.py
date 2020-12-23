import json
from collections import defaultdict

import jsonlines
import numpy as np
import torch
from mllib.decorators import get_duration
from mllib.processing.Label import process_answer_default
from mllib.utils import clip_img_ids, get_subfiles_from_path, load_arrow


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
    elif not config.dryrun and not config.data.img_first:
        stop_int = max(1, int(np.ceil(stop_int * config.data.percent_data)))

    print("attempt load ", stop_int, " data samples")
    text_data = []
    imgid_to_text = defaultdict(list)
    text_img_ids = set()
    data_stop = 0
    for stream in streams:
        for idx, data in enumerate(stream):
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
                else:
                    text_img_ids.add(img_id)
                text_data.append(entry)
            data_stop += 1

            if config.data.img_first and config.dryrun:
                if all(list(map(lambda l: len(l) >= 2, imgid_to_text.values()))):
                    break
            if stop_int == data_stop:
                break

    if config.data.img_first:
        img_ids = list(imgid_to_text.keys())
        if not config.dryrun:
            img_ids = clip_img_ids(img_ids, config.data.percent_data)
        else:
            assert len(img_ids) == bz, (len(img_ids), bz)
            check = list(map(lambda l: len(l) >= 2, imgid_to_text.values()))
            if data_stop != stop_int:
                assert all(check), (imgid_to_text, check)
    else:
        img_ids = text_img_ids

    assert len(text_data) > 0
    assert len(img_ids) > 0
    print(f"num ignored: {num_ignored} ignored: {ignored_labels}")
    return {"text": text_data, "pathes": path_dict, "arrow": arrow, "labels": labels, "img_ids": img_ids,
            "imgid_to_text": imgid_to_text}
