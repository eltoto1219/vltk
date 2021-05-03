import json
import os

import numpy as np
import torch
import torchvision.transforms.functional as FV
from pycocotools import mask as coco_mask
from skimage import measure
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


def imagepoints_to_polygon(points):
    img = imagepoints_to_mask(points)
    polygon = mask_to_polygon(img)
    return polygon


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


def mask_to_polygon(mask):
    contours = measure.find_contours(mask, 0.5)
    seg = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        seg.append(segmentation)
    return seg


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


# def resize_mask(mask, transforms_dict):
#     if "Resize" in transforms_dict:
#         return transforms_dict["Resize"](mask)
#     else:
#         return mask


def resize_binary_mask(array, img_size, pad_size=None):
    img_size = (img_size[0], img_size[1])
    if array.shape != img_size:
        array = FV.resize(array.unsqueeze(0), img_size).squeeze(0)
        return array
    else:
        return array


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
