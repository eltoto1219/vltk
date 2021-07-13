import torch
import vltk.vars as vltk
from PIL import Image
from vltk.features import Features
from vltk import adapters
from vltk.configs import VisionConfig
from vltk.utils.adapters import rescale_box


class FRCNN(adapters.VisnExtraction):

    # TODO: currently, this image preprocessing config is not correct
    default_processor = VisionConfig(
        **{
            "transforms": ["FromFile", "Resize", "ToTensor", "Normalize"],
            "size": 800,
            "max_size": 1333,
            "mode": Image.BILINEAR,
            "pad_value": 0.0,
            "mean": [102.9801 / 255, 115.9465 / 255, 122.7717 / 255],
            "std": [1.0, 1.0, 1.0],
        }
    )

    @staticmethod
    def setup():
        from vltk import compat
        from vltk.modeling.frcnn import FRCNN as FasterRCNN

        weights = "unc-nlp/frcnn-vg-finetuned"
        model_config = compat.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        return FasterRCNN.from_pretrained(weights, model_config), model_config

    @staticmethod
    def schema(max_detections=36, visual_dim=2048):
        return {
            "attr_ids": Features.Ids(),
            "object_ids": Features.Ids(),
            vltk.features: Features.Features3D(max_detections, visual_dim),
            vltk.box: Features.Box(),
        }

    @staticmethod
    def forward(model, entry):

        size = entry[vltk.size]
        scale_wh = entry[vltk.scale]
        image = entry[vltk.img]

        model_out = model(
            images=image.unsqueeze(0),
            image_shapes=size.unsqueeze(0),
            padding="max_detections",
            pad_value=0.0,
            location="cpu",
        )
        normalized_boxes = torch.round(rescale_box(model_out["boxes"][0], 1 / scale_wh))

        return {
            "object_ids": [model_out["obj_ids"][0].tolist()],
            "attr_ids": [model_out["attr_ids"][0].tolist()],
            vltk.box: [normalized_boxes.tolist()],
            vltk.features: [model_out["roi_features"][0]],
        }
