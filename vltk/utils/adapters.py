import json
import os

import numpy as np
from pycocotools import mask as coco_mask
from vltk.processing.data import Data
from vltk.processing.image import (Image, get_pad, get_rawsize, get_scale,
                                   get_size)
from vltk.processing.label import Label
from vltk.processing.optim import Optim
from vltk.processing.sched import Sched

PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "libdata"
)
ANS_CONVERT = json.load(open(os.path.join(PATH, "convert_answers.json")))
CONTRACTION_CONVERT = json.load(open(os.path.join(PATH, "convert_answers.json")))


def rescale_box(boxes, hw_scale):
    # boxes = (n, (x, y, w, h))
    # x = top left x position
    # y = top left y position
    h_scale = hw_scale[0]
    w_scale = hw_scale[1]
    y_centroids = (boxes[:, 1] - boxes[:, 3] / 2) * h_scale
    x_centroids = (boxes[:, 0] + boxes[:, 2] / 2) * w_scale
    boxes[:, 2] *= w_scale
    boxes[:, 3] *= h_scale
    boxes[:, 0] = x_centroids - boxes[:, 2] / 2  # scaled xs
    boxes[:, 1] = y_centroids + boxes[:, 3] / 2  # scaled ys
    return boxes


def seg_to_mask(segmentation, h, w):
    segmentation = coco_mask.decode(coco_mask.frPyObjects(segmentation, h, w))
    if len(segmentation.shape) < 3:
        segmentation = segmentation[..., None]
    segmentation = np.any(segmentation, axis=-1).astype(np.uint8)
    return segmentation


def resize_binary_mask(array, img_size, pad_size=None):
    img_size = tuple(img_size.tolist())
    if array.shape != img_size:

        raise Exception(array.shape)
        return bilinear_interpolate(array, *img_size)
        # need to implement
    else:
        return array


def uncompress_mask(compressed, size):
    mask = np.zeros(size, dtype=np.uint8)
    mask[compressed[0], compressed[1]] = 1
    return mask


def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


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
