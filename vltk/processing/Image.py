import os

import cv2
import torch
import torch.nn.functional as F


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
        img = (img - torch.tensor(mean).unsqueeze(-1).unsqueeze(-1)) / torch.tensor(sdev).unsqueeze(-1).unsqueeze(-1)
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
