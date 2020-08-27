import cv2
from collections import namedtuple
from PIL import Image
import np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

class ResizeShortestEdge:

    def __init__(
            self,
            short_edge_length,
            max_size=sys.maxsize,
            interp=Image.BILINEAR
        ):
            """
            Args:
                short_edge_length (list[min, max])
                max_size (int): maximum allowed longest edge length.
                sample_style (str): either "range" or "choice".
            """
            self.interp_method = interp
            self.max_size = max_size

    def __call__(self, img):
        h, w = img.shape[:2]
        size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
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
            pil_image = pil_image.resize((neww, newh), interp_method)
            ret = np.asarray(pil_image)
        else:
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            mode =  Image.BILINEAR
            img = F.interpolate(img, (newh, neww), mode=interp_method, align_corners=False)
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
        self.max_image_size = cfg.MAX_SIZE

    def __call__(self, img_jpg, input_format = "RGB"):
        # read file
        im = cv2.imread(img_jpg)
        # convert rgb
        img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # if self.input_format == "RGB":
        #    original_image = original_image[:, :, ::-1]
        height, width = img_rgb.shape[:2]
        # Preprocessing
        image = self.aug(img_rgb)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0,
            1)).to(self.device)
        #normalize the image values
        image = (image - self.pixel_mean) / self.pixel_std
        inputs = {"image": image, "height": height, "width": width}
        # Normalize
        # PAD:
        assert isinstance(image, torch.Tensor), type(image)
        # per dimension maximum (H, W) or (C_1, ..., C_K, H, W) where K >= 1 among all tensors
        max_size = max(image.shape)
        if self.stride > 0:
            import math
            stride = self.stride
            max_height = int(math.ceil(image.shape[-2] / stride) * stride)  # type: ignore
            max_width = int(math.ceil(image.shape[-1] / stride) * stride)  # type: ignore
            max_size = tuple([max_height, max_width])
        image_size = image.shape[-2:]
        padded = F.pad(
            image,
            [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]],
            value=self.pad_value,
        )
        image = padded.contiguous()
        new_height, new_width = image.shape[:2]
        scale_x = 1. * new_width / width
        scale_y = 1. * new_height / height
        return (image, (height, width), (scale_x, scale_y))


def post_process_img(img_processed, boxes=None, masks=None, keypoints=None,  mask_threshold=0.5):
    """Rescale and apply boxes
    """
    imgage, output_height, output_width = img_processed

    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])

    aux_viz = None
    if boxes is not None:
        aux_viz = "boxes"
    elif masks is not None:
        aux_viz = "masks"
    elif keypoints is not None:
        aux_viz = "keypoints"
    if aux_viz is not None:
        aux_viz.scale(scale_x, scale_y)
        aux_viz.clip(results.image_size)

    return img_processed






