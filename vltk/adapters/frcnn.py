import torch
import vltk
from vltk import Features, adapters
from vltk.configs import VisionConfig
from vltk.utils.adapters import rescale_box


class FRCNN(adapters.VisnExtraction):

    # TODO: currently, this image preprocessing config is not correct
    default_processor = VisionConfig(
        **{
            "transforms": ["FromFile", "ToTensor", "Resize", "Normalize"],
            "size": (800, 1333),
            "mode": "bilinear",
            "pad_value": 0.0,
            "mean": [102.9801 / 255, 115.9465 / 255, 122.7717 / 255],
            "std": [1.0, 1.0, 1.0],
        }
    )

    def setup():
        from vltk import compat
        from vltk.modeling.frcnn import FRCNN as FasterRCNN

        weights = "unc-nlp/frcnn-vg-finetuned"
        model_config = compat.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        return FasterRCNN.from_pretrained(weights, model_config), model_config

    def schema(max_detections=36, visual_dim=2048):
        return {
            "attr_ids": Features.Ids,
            "object_ids": Features.Ids,
            vltk.features: Features.Features3D(max_detections, visual_dim),
            vltk.box: Features.Box,
        }

    def forward(model, entry):

        size = entry["size"]
        scale_hw = entry["scale"]
        image = entry["image"]

        model_out = model(
            images=image.unsqueeze(0),
            image_shapes=size.unsqueeze(0),
            scales_yx=scale_hw.unsqueeze(0),
            padding="max_detections",
            pad_value=0.0,
            location="cpu",
        )
        normalized_boxes = torch.round(
            rescale_box(model_out["boxes"][0], 1 / entry["scale"])
        )

        return {
            "object_ids": [model_out["obj_ids"][0].tolist()],
            "attr_ids": [model_out["attr_ids"][0].tolist()],
            vltk.box: [normalized_boxes.tolist()],
            vltk.features: [model_out["roi_features"][0]],
        }
