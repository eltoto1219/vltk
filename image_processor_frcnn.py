import sys
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class ResizeShortestEdge:
    def __init__(self, short_edge_length, max_size=sys.maxsize, interp=Image.BILINEAR):
        """
        Args:
            short_edge_length (list[min, max])
            max_size (int): maximum allowed longest edge length.
        """
        self.interp_method = interp
        self.max_size = max_size
        self.short_edge_length = short_edge_length

    def __call__(self, img):
        h, w = img.shape[:2]
        # later: provide list and randomly choose index for resize
        size = np.random.randint(
            self.short_edge_length[0], self.short_edge_length[1] + 1
        )
        if size == 0:
            return img
        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        if img.dtype == np.uint8:
            pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((neww, newh), self.interp_method)
            ret = np.asarray(pil_image)
        else:
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            img = F.interpolate(
                img, (newh, neww), mode=self.interp_method, align_corners=False
            )
            shape[:2] = (newh, neww)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)
        return ret


class PreProcess:
    def __init__(self, cfg):
        self.aug = ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.input_format = cfg.INPUT.FORMAT
        self.size_divisibility = cfg.SIZE_DIVISIBILITY
        self.pad_value = cfg.PAD_VALUE
        self.max_image_size = cfg.INPUT.MAX_SIZE_TEST
        self.device = cfg.MODEL.DEVICE
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device)
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device)

    def __call__(self, img_tensor, max_size=None, input_format="RGB"):
        """Args:
        max_size (tuple): [mas_height, max_width], should pad to batch_wise_max, else:
        defualts to max_size set in config
        """
        if max_size is None:
            max_size = (
                self.max_image_size,
                self.max_image_size,
            )
        else:
            max_size = (
                min(self.max_image_size, max_size[0]),
                min(self.max_image_size, max_size[1]),
            )
        with torch.no_grad():
            img_rgb = img_tensor
            if self.input_format == "RGB":
                img_rgb = img_rgb[:, :, ::-1]
            height, width = img_rgb.shape[:2]
            image_hw = torch.Tensor([height, width])
            # Preprocessing
            image = self.aug(img_rgb)
            scaled_hw = torch.Tensor(image.shape[:2])
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(
                self.device
            )
            # Normalize the image values
            image = image - self.pixel_mean.unsqueeze(-1).unsqueeze(
                -1
            ) / self.pixel_std.unsqueeze(-1).unsqueeze(-1)
            # PAD:
            if self.size_divisibility > 0:
                raise NotImplementedError()
                """
                stride = self.size_divisibility
                import math

                stride = self.stride
                # type: ignore
                max_height = int(math.ceil(image.shape[-2] / stride) * stride)
                # type: ignore
                max_width = int(math.ceil(image.shape[-1] / stride) * stride)
                max_size = tuple([max_height, max_width])
                """
            image_size = image.shape[-2:]
            # pad vals: (pl, pr, pl, pr)
            pad_vals = (
                0,
                max_size[0] - image_size[1],
                0,
                max_size[1] - image_size[0],
            )
            padded = F.pad(
                image,
                pad_vals,
                value=self.pad_value,
            )
            image = padded.contiguous()
            new_height, new_width = image.shape[-2], image.shape[-1]
            scale_x = 1.0 * new_width / scaled_hw[0]
            scale_y = 1.0 * new_height / scaled_hw[1]
            return (image, image_hw, torch.Tensor([scale_x, scale_y]))


def postprocess(boxes, input_hw, output_hw):
    scale_x, scale_y = (
        output_hw[1] / input_hw[1],
        output_hw[0] / input_hw[0],
    )
    boxes[:, 0::2] *= scale_x
    boxes[:, 1::2] *= scale_y
    return boxes.int()


def tensorize(im):
    im = cv2.imread(im)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _clip_box(tensor, box_size: Tuple[int, int]):
    assert torch.isfinite(tensor).all(), "Box tensor contains infinite or NaN!"
    h, w = box_size
    tensor[:, 0].clamp_(min=0, max=w)
    tensor[:, 1].clamp_(min=0, max=h)
    tensor[:, 2].clamp_(min=0, max=w)
    tensor[:, 3].clamp_(min=0, max=h)
