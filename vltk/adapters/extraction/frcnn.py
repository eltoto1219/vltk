import datasets as ds
import torch
from vltk.abc.imageset import Imageset

TESTPATH = "/home/avmendoz/vltk/tests"


# user must only define forward in this function


class FRCNNSet(Imageset):
    # name will be overwritten with the name of the dataset when loaded from file
    name = "frcnn"

    @staticmethod
    def default_features(max_detections, pos_dim, visual_dim):
        return {
            "attr_ids": ds.Sequence(length=max_detections, feature=ds.Value("float32")),
            "attr_probs": ds.Sequence(
                length=max_detections, feature=ds.Value("float32")
            ),
        }

    def forward(filepath, image_preprocessor, model, **kwargs):

        pad_value = kwargs.get("pad_value", 0.0)
        min_size = kwargs.get("min_size", 800)
        max_size = kwargs.get("max_size", 1333)
        pxl_mean = kwargs.get("pxl_mean", [102.9801, 115.9465, 122.7717])
        pxl_sdev = kwargs.get("pxl_sdev", [1.0, 1.0, 1.0])
        device = kwargs.get("device", "cpu")

        image, sizes, scale_hw = image_preprocessor(
            filepath,
            min_size=min_size,
            max_size=max_size,
            mean=pxl_mean,
            sdev=pxl_sdev,
        )

        sizes = torch.tensor(list(sizes))
        scale_hw = torch.tensor(list(scale_hw))

        image, sizes, scale_hw = (
            image.to(torch.device(device)),
            sizes.to(torch.device(device)),
            scale_hw.to(torch.device(device)),
        )

        output_dict = model(
            images=image.unsqueeze(0),
            image_shapes=sizes.unsqueeze(0),
            scales_yx=scale_hw.unsqueeze(0),
            padding="max_detections",
            pad_value=pad_value,
            return_tensors="np",
            location="cpu",
        )

        return output_dict


# frcnnconfig = compat.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
# frcnnconfig.model.device = 0
# frcnn = FRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnnconfig)
# config = Config().data
