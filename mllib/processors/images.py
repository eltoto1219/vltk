import os

import cv2
import torch
import torch.nn.functional as F


def img_to_tensor(
    fp,
    min_size=2160,
    max_size=2160,
    pad_value=None,
    mean=None,
    sdev=None,
):
    assert isinstance(fp, str)
    if isinstance(mean, list):
        mean = torch.tensor(mean).view(-1, 1, 1)
    else:
        assert isinstance(mean, torch.tensor), type(mean)
        assert mean.shape == (3, 1, 1), mean.shape
    if isinstance(sdev, list):
        sdev = torch.tensor(sdev).view(-1, 1, 1)
    else:
        assert isinstance(sdev, torch.tensor), type(sdev)
        assert sdev.shape == (3, 1, 1), sdev.shape
    assert os.path.isfile(fp), fp
    img = cv2.imread(fp)
    if img is None:
        return None, None, None
    img = img[:, :, ::1]
    img = torch.as_tensor(img).float()
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
    img = img.permute(2, 0, 1).unsqueeze(0)
    img = F.interpolate(
        img, (newh, neww), mode="bilinear", align_corners=False
    ).squeeze(0)
    img = torch.clamp(img, max=255)
    if mean is not None and sdev is not None:
        img = (img - mean) / sdev
    if pad_value is not None:
        size = img.shape[-2:]
        img = F.pad(
            img,
            [0, max_size - size[1], 0, max_size - size[0]],
            value=pad_value,
        )
    img = img.unsqueeze(0)
    sizes = torch.tensor([h, w]).unsqueeze(0)
    raw_sizes = torch.tensor([newh, neww]).unsqueeze(0)
    scales_hw = torch.true_divide(raw_sizes, sizes)
    return img, raw_sizes, scales_hw
