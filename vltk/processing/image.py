import os

import cv2
import torch
import torch.nn.functional as F
from vltk import IMAGEPROCPATH
from vltk.inspect import import_funcs_from_file


class Image:
    def __init__(self):
        if "IMAGEPROCDICT" not in globals():
            global IMAGEPROCDICT
            IMAGEPROCDICT = import_funcs_from_file(IMAGEPROCPATH, pkg="vltk.processing")

    def avail(self):
        return list(IMAGEPROCDICT.keys())

    def get(self, name):
        return IMAGEPROCDICT[name]

    def add(self, name, proc):
        IMAGEPROCDICT[name] = proc


def resize_short_edge(
    img,
    order="chw",
    min_size=512,
    max_size=768,
    mode="bicubic",
    scale=True,
    pad=True,
    gpu=False,
    square=False,
):
    if gpu:
        img = img.cuda()
    with torch.no_grad():
        orig_sizes = []
        sizes = []
        if order == "chw":
            C, H, W = img.shape
        elif order == "hwc":
            H, W, C = img.shape
            img = img.permute(3, 1, 2)

        scale = min_size * 1.0 / min(H, W)

        if H < W:
            newh, neww = min_size, scale * W
        else:
            newh, neww = scale * H, min_size

        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale

        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        img = img.unsqueeze(0)
        if not square:
            img = F.interpolate(img, (newh, neww), mode=mode, align_corners=False)
            img = img.squeeze(0)
        else:
            img = F.interpolate(
                img, (max_size, max_size), mode=mode, align_corners=False
            )
            img = img.squeeze(0)
        if scale:
            img /= 255
        if not square and pad:
            img = F.pad(img, [0, max_size - neww, 0, max_size - newh], value=0)
        orig_sizes.append([H, W])
        sizes.append([newh, neww])

        orig_sizes = torch.tensor(orig_sizes)
        sizes = torch.tensor(sizes)

    return {"imgs": img, "sizes": sizes, "orig_sizes": orig_sizes}


def img_to_tensor(
    fp, min_size=832, max_size=832, pad_value=None, mean=None, sdev=None, use_gpu=False
):
    assert isinstance(fp, str)
    assert os.path.isfile(fp)
    img = cv2.imread(fp)
    if img is None:
        return None, (None, None), (None, None)
    img = img[:, :, ::1]
    img = torch.as_tensor(img).float()
    if use_gpu:
        img = img.cuda()
    h, w = img.shape[:2]
    scale = min_size * 1.0 / min(h, w)
    if h < w:
        newh, neww = min_size, scale * w
    else:
        newh, neww = scale * h, min_size
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)

    img = img.permute(2, 0, 1).unsqueeze(0)  # 3, 0, 1)  # hw(c) -> nchw
    img = F.interpolate(
        img, (newh, neww), mode="bilinear", align_corners=False
    ).squeeze(0)
    img = torch.clamp(img, max=255)
    if mean is not None and sdev is not None:
        img = (img - torch.tensor(mean).unsqueeze(-1).unsqueeze(-1)) / torch.tensor(
            sdev
        ).unsqueeze(-1).unsqueeze(-1)
    if pad_value is not None:
        size = img.shape[-2:]
        img = F.pad(
            img,
            [0, max_size - size[1], 0, max_size - size[0]],
            value=pad_value,
        )

    sizes = torch.tensor([newh, neww])
    raw_sizes = torch.tensor([h, w])
    scales_hw = torch.true_divide(raw_sizes, sizes)
    del raw_sizes
    del sizes

    if use_gpu:
        img = img.cpu()
        scales_hw = scales_hw.cpu()

    return img, (h, w), scales_hw
