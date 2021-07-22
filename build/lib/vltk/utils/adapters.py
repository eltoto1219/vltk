import json
import os
from collections import defaultdict

import numpy as np
import torch
import vltk.vars as vltk
from matplotlib.patches import Rectangle
from PIL import Image
from pycocotools import mask as coco_mask
from torchvision.transforms.functional import resize
from tqdm import tqdm
from vltk.processing.image import get_pad, get_rawsize, get_scale, get_size

PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "libdata"
)
ANS_CONVERT = json.load(open(os.path.join(PATH, "convert_answers.json")))
CONTRACTION_CONVERT = json.load(open(os.path.join(PATH, "convert_answers.json")))


def map_ocr_predictions(pred, tokenmap, gold=None, boxes=None, ignore_id=-100):
    golds = []
    preds = []
    accs = []
    if gold is not None:
        for g, t, p in zip(gold, tokenmap, pred):
            t = t[: t.argmin()]
            total = 0
            for i, v in enumerate(t):
                total += v
                if total >= len(g):
                    break
            t = t[:i]

            tsum = sum(t)
            split_g = torch.split(g[:tsum], t.cpu().tolist())
            split_p = torch.split(p[:tsum], t.cpu().tolist())
            true_gold = torch.stack([x[0] for x in split_g]).cpu().tolist()
            true_preds = torch.stack([x.mode().values for x in split_p]).cpu().tolist()
            for i in reversed(range(len(true_gold))):
                if true_gold[i] == -100:
                    true_gold.pop(i)
                    true_preds.pop(i)
            accs += [(1 if p == g else 0) for p, g in zip(true_preds, true_gold)]
            preds += true_preds
            golds += true_gold
        return golds, preds, accs
    else:
        for t, p in zip(tokenmap, pred):
            t = t[: t.argmin()]
            total = 0
            for i, v in enumerate(t):
                total += v
                if total >= len(p):
                    break
            t = t[:i]

            tsum = sum(t)
            split_p = torch.split(p[:tsum], t.cpu().tolist())
            true_preds = torch.stack([x.mode().values for x in split_p]).cpu().tolist()
            preds += true_preds
        bboxes = None
        if boxes is not None:
            bboxes = []
            for t, b in zip(tokenmap, boxes):
                t = t[: t.argmin()]
                total = 0
                for i, v in enumerate(t):
                    total += v
                    if total >= len(b):
                        break
                t = t[:i]

                tsum = sum(t)
                split_b = torch.split(b[:tsum], t.cpu().tolist())
                temp_boxes = []
                for s in split_b:
                    temp_boxes.append(s.tolist()[0])
                true_boxes = temp_boxes
                bboxes += true_boxes
        return preds, bboxes


def histogram_from_counter(counter, truncate_labs=False, min_freq=0, x_label=""):
    # credits to stack overflow
    # TODO: add credits to stack overflow for other functions that I am using form there
    import matplotlib.pyplot as plt

    alt_msg = ""
    if "None" in counter:
        n_none = counter["None"]
        alt_msg = f"None: {n_none}"
        counter.pop("None")
    counter = sorted(list(counter.items()), key=lambda x: x[1])
    n_bins = len(counter)
    for i, (k, v) in enumerate(counter):
        if v >= min_freq:
            break
    counter = counter[i:]

    labels, values = zip(*counter)
    if truncate_labs:
        labels = [labels[0]] + ["" for x in labels[:-2]] + [labels[-1]]
    indexes = np.arange(len(labels))
    width = 1
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels, rotation=-10)
    plt.ylabel("counts")
    plt.xlabel(x_label)
    plt.tight_layout()
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0)
    plt.legend(
        [
            extra,
        ],
        (f"n_bins: {n_bins}; min_freq: {min_freq}" + f"; {alt_msg}",),
    )
    plt.show()


# def imagepoints_to_polygon(points):
#     img = imagepoints_to_mask(points)
#     polygon = mask_to_polygon(img)
#     return polygon


# source: https://github.com/ksrath0re/clevr-refplus-rec/
def imagepoints_to_mask(points, size):
    # raise Exception(points)
    # npimg = []
    img = []
    cur = 0
    for num in points:
        # if cur == 0:
        #     part = np.zeros(int(num))
        # else:
        #     part = np.concatenate((part, np.ones(int(num))))
        # npimg.append(part)
        num = int(num)
        img += [cur] * num
        cur = 1 - cur
    img = torch.tensor(img).reshape(tuple(size.tolist()))
    # npimg = np.stack(npimg)
    # raise Exception(part.shape)
    # part = part.reshape(tuple(size.tolist()))
    return img


# def mask_to_polygon(mask):
#     contours = measure.find_contours(mask, 0.5)
#     seg = []
#     for contour in contours:
#         contour = np.flip(contour, axis=1)
#         segmentation = contour.ravel().tolist()
#         seg.append(segmentation)
#     return seg


def rescale_box(boxes, wh_scale):
    # boxes = (n, (x, y, w, h))
    # x = top left x position
    # y = top left y position
    h_scale = wh_scale[1]
    w_scale = wh_scale[0]
    boxes[:, 0] *= w_scale
    boxes[:, 1] *= h_scale
    boxes[:, 2] *= w_scale
    boxes[:, 3] *= h_scale

    return boxes


def seg_to_mask(segmentation, w, h):
    segmentation = coco_mask.decode(coco_mask.frPyObjects(segmentation, h, w))
    if len(segmentation.shape) < 3:
        segmentation = segmentation[..., None]
    segmentation = np.any(segmentation, axis=-1).astype(np.uint8)
    return torch.from_numpy(segmentation).bool()


# def resize_mask(mask, transforms_dict):
#     if "Resize" in transforms_dict:
#         return transforms_dict["Resize"](mask)
#     else:
#         return mask


def resize_binary_mask(array, img_size, pad_size=None):
    img_size = (img_size[0], img_size[1])
    # raise Exception(img_size)
    if array.shape != img_size:
        if array.dim() == 2:
            array = array.unsqueeze(0).unsqueeze(0)
        else:
            array = array.unsqueeze(0)

        array = resize(array, img_size)

        if array.dim() == 2:
            array = array.squeeze(0).squeeze(0)
        else:
            array = array.squeeze(0)

        return array
    else:
        return torch.from_numpy(array)


def uncompress_mask(compressed, size):
    mask = np.zeros(size, dtype=np.uint8)
    mask[compressed[0], compressed[1]] = 1
    return mask


def clean_label(ans):
    if len(ans) == 0:
        return ""
    ans = ans.lower()
    ans = ans.replace(",", "")
    if ans[-1] == ".":
        ans = ans[:-1].strip()
    if ans.startswith("a "):
        ans = ans[2:].strip()
    if ans.startswith("an "):
        ans = ans[3:].strip()
    if ans.startswith("the "):
        ans = ans[4:].strip()
    ans = " ".join(
        [
            CONTRACTION_CONVERT[a] if a in CONTRACTION_CONVERT else a
            for a in ans.split(" ")
        ]
    )
    if ans in ANS_CONVERT:
        ans = ANS_CONVERT[ans]
    return ans


def soft_score(occurences):
    if occurences == 0:
        return 0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1


def get_span_via_jaccard(words, answers, skipped=None):
    """
    inputs:
        words: tuple of strings (each string == one word)
        answers: list of strings (each string == one word or many words)
        skipped: int or None
    outputs:
        span: list of length `len(words)` of 1's and 0's. 0: is not apart of answer, 1: is apart of answer
        max_jaccard: the similarity metric value 0.0-1.0 for how well answer matched in span
    """
    span = [0] * len(words)
    any_ans = False
    for ans in answers:
        # single word case
        if len(ans.split()) == 1:
            try:
                idx = words.index(ans.lower())
            except Exception:
                continue
            if idx is not None:
                span[idx] = 1
                max_jaccard = 1.0
                any_ans = True
                break

    if not any_ans:
        keep = None
        max_jaccard = -0.1
        for ans in answers:
            if len(ans.split()) == 1:
                for ans in answers:
                    ans = set(ans.lower())
                    for idx, word in enumerate(words):
                        word = set(word.lower())
                        jaccard = len(word & ans) / len(word | ans)
                        if jaccard > max_jaccard:
                            max_jaccard = jaccard
                            keep = idx
            else:
                end_keep = len(words)

                ans = ans.split()
                start_keep = end_keep - len(ans)
                start = ans[0].lower()
                end = ans[-1].lower()
                start = set(start)
                end = set(end)
                jaccards = []
                for idx, word in enumerate(words[: -len(ans)]):
                    temp_jaccard = 0.0
                    for jdx, subans in enumerate(ans):
                        word = set(words[idx + jdx].lower())
                        subans = set(subans)
                        temp_jaccard += len(word & subans) / len(word | subans)
                        jaccards.append(
                            (
                                temp_jaccard / len(ans),
                                (idx, idx + len(ans)),
                            )
                        )
                if not jaccards:
                    continue
                jaccard, (start_keep, end_keep) = sorted(jaccards, key=lambda x: x[0])[
                    -1
                ]
                if jaccard > max_jaccard:
                    keep = (start_keep, end_keep)
                    span[start_keep:end_keep] = [1] * ((end_keep - start_keep) + 1)
                    max_jaccard = jaccard

        if keep is None:
            span = None
        elif isinstance(keep, tuple):
            span[keep[0] : keep[1]] = [1] * ((keep[1] - keep[0]) + 1)
        else:
            span[keep] = 1

    if max_jaccard == 0.0:
        if skipped is not None:
            skipped += 1
        span = None

    return span, max_jaccard


def truncate_and_pad_list(inp_list, max_len, pad_value=""):
    inp_list = inp_list[: min(max_len, len(inp_list))]
    inp_list += [pad_value] * (max_len - len(inp_list))
    return inp_list


def basic_coco_annotations(json_files, splits):
    """
    inputs:
        json_files: a dict of annotation files in  coco format -->
            keys: filename
            values: json file
            ===
            boxes in (x,y,w,h) format
            segmentations are polygons
            ...
        splits: list of respective splits aligned to the keys of json_files
    outputs:
        list of dictionaries:
            keys:
                vltk.imgid
                vltk.box
                vltk.polygons
                vltk.objects
            values:
                str: the image id (stem of the filename)
                list of list of floats: list of bounding boxes
                list of list of list of floats: list of polygons
                list of strings: respective label classes for each object/segmentation
    """
    total_annos = {}
    id_to_cat = {}
    file_to_id_to_stem = defaultdict(dict)
    for file, data in sorted(json_files.items(), key=lambda x: x[0]):
        info = data["images"]
        for i in info:
            img_id = i["file_name"].split(".")[0]

            file_to_id_to_stem[file][i["id"]] = img_id

        # for file, data in tqdm(sorted(json_files.items(), key=lambda x: x[0])):
        categories = data["categories"]
        for cat in categories:
            id_to_cat[cat["id"]] = cat["name"]

        for entry in data["annotations"]:

            img_id = str(file_to_id_to_stem[file][entry["image_id"]])
            bbox = entry["bbox"]
            segmentation = entry["segmentation"]
            category_id = id_to_cat[entry["category_id"]]
            if entry["iscrowd"]:
                seg_mask = []
            else:
                seg_mask = segmentation
                if not isinstance(seg_mask[0], list):
                    seg_mask = [seg_mask]
            img_data = total_annos.get(img_id, None)
            if img_data is None:
                img_entry = defaultdict(list)
                img_entry[vltk.objects].append(category_id)
                img_entry[vltk.box].append(bbox)
                img_entry[vltk.polygons].append(seg_mask)
                total_annos[img_id] = img_entry
            else:
                total_annos[img_id][vltk.box].append(bbox)
                total_annos[img_id][vltk.objects].append(category_id)
                total_annos[img_id][vltk.polygons].append(seg_mask)

    return [{vltk.imgid: img_id, **entry} for img_id, entry in total_annos.items()]
