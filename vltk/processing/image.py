from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FV
from PIL import Image as PImage
from torchvision import transforms


def get_scale(obj):
    if not hasattr(obj, "transforms"):
        return None
    scale = None
    for t in obj.transforms:
        if hasattr(t, "_scale"):
            scale = t._scale
    return scale


def get_size(obj):
    if not hasattr(obj, "transforms"):
        return None
    size = None
    for t in obj.transforms:
        if hasattr(t, "_size"):
            size = t._size
    return size


def get_rawsize(obj):
    if not hasattr(obj, "transforms"):
        return None
    size = None
    for t in obj.transforms:
        if hasattr(t, "_rawsize"):
            size = t._rawsize
    return size


class ToPILImage(object):
    def __init__(self, mode=None):
        self.mode = mode
        pass

    def __call__(self, filepath):
        if isinstance(filepath, str):
            return PImage.open(filepath)
        else:
            return FV.to_pil_image(filepath.byte(), self.mode)


class ToTensor(object):
    def __init__(self):
        self._scale = torch.tensor([1.0, 1.0])
        self._size = None
        self._rawsize = None
        pass

    def __call__(self, pilimg):
        nump = np.array(pilimg)
        tensor = torch.as_tensor(
            nump.reshape(nump.shape[-1], *nump.shape[:2]), dtype=torch.float
        ).float()
        nump = np.array(tensor.shape)

        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)

        self._rawsize = torch.tensor(tensor.shape[1:])
        self._size = torch.tensor(tensor.shape[1:])

        return tensor


class Normalize(object):
    def __init__(self, mean=None, std=None, inplace=False, scale="standard"):
        self.mean = mean
        self.std = std
        self.scale = scale
        self.inplace = inplace
        self._std = None
        self._mean = None

    def __call__(self, tensor):
        # tensor must be: (C, H, W)
        if self.mean is None or self.std is None:
            if self.scale == "standard":
                mean = tensor.mean(dim=(-1, -2))
                std = torch.sqrt((tensor - mean.reshape(-1, 1, 1)) ** 2).mean(
                    dim=(-1, -2)
                )
                mean = mean.tolist()
                std = std.tolist()
                self._std = std
                self._mean = mean
            else:
                return tensor / 255
        else:
            mean = self.mean
            std = self.std

        normalize = FV.normalize(tensor, mean, std, self.inplace)
        return normalize


class ResizeTensor(object):
    def __init__(self, size=(512, 768), mode="bicubic", gpu=None, aspect_ratio=False):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        if isinstance(size, int):
            min_size = size
            max_size = size
        else:
            min_size = min(size)
            max_size = max(size)

        self.size = size
        self.mode = mode
        self.gpu = gpu
        self.max_size = max_size
        self.min_size = min_size
        self.aspect_ratio = aspect_ratio
        self._rawsize = None
        self._size = None
        self._scale = None

    def __scale(self):
        if self._size is not None and self._rawsize is not None:
            with torch.no_grad():
                return torch.tensor(
                    [self._size[0] / self._rawsize[0], self._size[1] / self._rawsize[1]]
                )
        else:
            return None

    @torch.no_grad()
    def __call__(self, tensor):
        max_size = self.max_size
        min_size = self.min_size
        C, H, W = tensor.shape
        self._rawsize = torch.tensor(tensor.shape[1:])
        tensor = tensor.unsqueeze(0)
        if self.gpu is not None:
            tensor = tensor.to(torch.device(self.gpu))
        if self.min_size != self.max_size:
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

            tensor = F.interpolate(
                tensor, (newh, neww), mode=self.mode, align_corners=False
            ).squeeze(0)
            self._size = torch.tensor(tensor.shape[1:])
            tensor = torch.clamp(tensor, max=255)
            self._scale = self.__scale()
            return tensor
        else:
            tensor = F.interpolate(
                tensor, (min_size, max_size), mode=self.mode, align_corners=False
            ).squeeze(0)
            self._size = torch.tensor(tensor.shape[1:])
            tensor = torch.clamp(tensor, max=255)
            self._scale = self.__scale()
            return tensor


class Pad(object):
    def __init__(self, size=768, pad_value=0.0):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        if isinstance(size, int):
            max_size = size
        else:
            max_size = max(size)
        self.size = size
        self.pad_value = pad_value
        self.max_size = max_size
        self._size = None

    @torch.no_grad()
    def __call__(self, tensor):
        C, H, W = tensor.shape
        max_size = self.max_size
        tensor = F.pad(tensor, [0, max_size - W, 0, max_size - H], value=self.pad_value)
        self._size = torch.tensor(tensor.shape[1:])
        return tensor


# def Pipeline(
#     size=768,
#     mode="bicubic",
#     scale="standard",
#     gpu=None,
#     pad_value=0,
#     std=None,
#     mean=None,
#     inplace=True,
#     pad=True,
#     resize=True,
#     normalize=True,
#     **kwargs,
# ):
#     process = [ToPILImage(), ToTensor()]

#     if resize:
#         process.append(ResizeTensor(size=size, mode=mode, gpu=gpu))

#     if normalize:
#         process.append(Normalize(mean=mean, std=std, inplace=inplace, scale=scale))

#     if not isinstance(size, int) and min(size) != max(size) and pad:
#         process.append(Pad(size=size, pad_value=pad_value))

#     return transforms.Compose(process)


class Image:
    def __init__(self):
        if "IMAGEPROCDICT" not in globals():
            global IMAGEPROCDICT
            IMAGEPROCDICT = {}
            IMAGEPROCDICT["ToTensor"] = ToTensor
            IMAGEPROCDICT["ToPILImage"] = ToPILImage
            IMAGEPROCDICT["Pad"] = Pad
            IMAGEPROCDICT["ResizeTensor"] = ResizeTensor
            IMAGEPROCDICT["Normalize"] = Normalize

    def avail(self):
        return list(IMAGEPROCDICT.keys())

    def get(self, name):
        return IMAGEPROCDICT[name]

    def add(self, name, proc):
        IMAGEPROCDICT[name] = proc
