import inspect
import sys
from collections.abc import Iterable

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FV
from PIL import Image as PImage
from torchvision.transforms import transforms


def get_scale(obj):
    if not hasattr(obj, "transforms"):
        return None
    scale = None
    for t in obj.transforms:
        if hasattr(t, "_scale"):
            scale = t._scale
    return scale


def get_pad(obj):
    if not hasattr(obj, "transforms"):
        return None
    pad = None
    for t in obj.transforms:
        if hasattr(t, "_pad"):
            pad = t._pad
    return pad


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


class FromFile(object):
    _scale = torch.tensor([1.0, 1.0])
    _size = None
    _rawsize = None

    def __init__(self, mode=None, grayscale=False):
        self.mode = mode
        self.grayscale = grayscale
        pass

    def __call__(self, filepath):
        if isinstance(filepath, str):
            if not self.grayscale:
                img = PImage.open(filepath).convert("RGB")
            else:
                img = PImage.open(filepath).convert("L")
            self._size = torch.tensor(img.size)
            self._rawsize = self._size
            return img
        else:
            img = FV.to_pil_image(filepath.byte(), self.mode).convert("RGB")
            return img


class ToTensor(transforms.ToTensor):
    def __call__(self, pil):
        tensor = super().__call__(pil)
        return tensor


class Normalize(transforms.Normalize):
    _std = None
    _mean = None

    def __init__(self, mean=None, std=None, inplace=False):
        super().__init__(mean, std, inplace)

    def __call__(self, tensor):
        # tensor must be: (C, H, W)
        if self.mean is None or self.std is None:
            mean = tensor.mean(dim=(-1, -2))
            std = torch.sqrt((tensor - mean.reshape(-1, 1, 1)) ** 2).mean(dim=(-1, -2))
            mean = mean.tolist()
            std = std.tolist()
            self._std = std
            self._mean = mean
        else:
            return super().__call__(tensor)


# class Pad(object):
#     def __init__(self, size=768, pad_value=0.0):
#         assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
#         if isinstance(size, int):
#             max_size = size
#         else:
#             max_size = max(size)
#         self.size = size
#         self.pad_value = pad_value
#         self.max_size = max_size
#         self._size = None

#     @torch.no_grad()
#     def __call__(self, tensor):
#         C, H, W = tensor.shape
#         max_size = self.max_size
#         tensor = F.pad(tensor, [0, max_size - W, 0, max_size - H], value=self.pad_value)
#         self._size = torch.tensor(tensor.shape[1:])
#         return tensor


class Resize(transforms.Resize):
    _size = None
    _rawsize = None
    _scale = None

    def __scale(self):
        if self._size is not None and self._rawsize is not None:
            with torch.no_grad():
                return torch.tensor(
                    [self._size[0] / self._rawsize[0], self._size[1] / self._rawsize[1]]
                )
        else:
            return None

    def __call__(self, pilimg):
        # raise Exception(pilimg.shape)
        self._rawsize = torch.tensor(pilimg.size)
        pilimg = super().__call__(pilimg)
        self._size = torch.tensor(pilimg.size)
        self._scale = self.__scale()
        # raise Exception(self._rawsize, self._size, self._scale)

        return pilimg


class Pad(transforms.Pad):
    # need to implement the amount that the image is padded such that I can resize the
    # binary mask as needed
    _size = None
    _pad = None

    @torch.no_grad()
    def __call__(self, tensor):
        tensor = super().__call__(tensor)
        self._size = torch.tensor(tensor.shape[1:])
        return tensor


class Image:
    def __init__(self):
        if "IMAGEPROCDICT" not in globals():
            global IMAGEPROCDICT
            IMAGEPROCDICT = {
                m[0]: m[1]
                for m in inspect.getmembers(
                    sys.modules["torchvision.transforms.transforms"], inspect.isclass
                )
            }
            IMAGEPROCDICT["ToTensor"] = ToTensor
            IMAGEPROCDICT["FromFile"] = FromFile
            IMAGEPROCDICT["Pad"] = Pad
            IMAGEPROCDICT["Resize"] = Resize
            IMAGEPROCDICT["Normalize"] = Normalize

    def avail(self):
        return list(IMAGEPROCDICT.keys())

    def get(self, name):
        return IMAGEPROCDICT[name]

    def add(self, name, proc):
        IMAGEPROCDICT[name] = proc
