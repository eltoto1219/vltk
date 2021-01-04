import json

import pyarrow as pa
import torch
from mllib.abc.imageset import Imageset
from mllib.compat import Config
from mllib.modeling.frcnn import FRCNN

TESTPATH = "/home/avmendoz/mllib/tests"


# user must only define forward in this function


class FRCNNSet(Imageset):
    def forward(filepath, image_preprocessor, model, **kwargs):
        pad_value = kwargs.get("pad_value", 0.0)
        min_size = kwargs.get("min_size", 800)
        max_size = kwargs.get("max_size", 800)
        pxl_mean = kwargs.get("pxl_mean", None)
        pxl_sdev = kwargs.get("pxl_sdev", None)
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


if __name__ == "__main__":

    config = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    config.model.device = 0
    frcnn = FRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=config)
    imageset = FRCNNSet.extract(
        path=TESTPATH,
        model=frcnn,
        img_format="jpg",
        image_preprocessor="img_to_tensor",
        features="frcnn",
        max_detections=config.max_detections,
        pos_dim=4,
        visual_dim=2048,
        device=config.model.device,
        save_to="test.arrow",
    )

    loaded = FRCNNSet.from_file("test.arrow")
    imap = loaded.img_to_row_map
    imgid = next(iter(imap.keys()))
    print(imap)
    print("is aligned?", loaded.check_imgid_alignment())
    print(f"entry for {imgid}: ", loaded.get_image(imgid).keys())

    # load specific split:
    loaded = FRCNNSet.from_file("test.arrow", split="subdir_extract_2")
    print(loaded)
