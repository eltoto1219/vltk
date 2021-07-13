import vltk.vars as vltk
from vltk.features import Features
from vltk import compat
from vltk.adapters import VisnExtraction
from vltk.configs import ProcessorConfig
from vltk.modeling.frcnn import FRCNN as FasterRCNN


class FRCNN(VisnExtraction):

    default_processor = ProcessorConfig(
        **{
            "transforms": ["ToPILImage", "ToTensor", "ResizeTensor", "Normalize"],
            "size": (800, 1333),
            "mode": "bilinear",
            "pad_value": 0.0,
            "mean": [102.9801, 115.9465, 122.7717],
            "sdev": [1.0, 1.0, 1.0],
        }
    )
    # model_config = compat.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    weights = "unc-nlp/frcnn-vg-finetuned"
    model = FasterRCNN

    def schema(max_detections=36, visual_dim=2048):
        return {
            "attr_ids": Features.ids,
            "object_ids": Features.ids,
            vltk.features: Features.features(max_detections, visual_dim),
            vltk.box: Features.boxtensor(max_detections),
        }

    def forward(model, entry, **kwargs):

        size = entry["size"]
        scale_hw = entry["scale"]
        image = entry["image"]

        model_out = model(
            images=image.unsqueeze(0),
            image_shapes=size.unsqueeze(0),
            scales_yx=scale_hw.unsqueeze(0),
            padding="max_detections",
            pad_value=0.0,
            return_tensors="np",
            location="cpu",
        )
        return {
            "object_ids": model_out["obj_ids"],
            "attr_ids": model_out["attr_ids"],
            vltk.box: model_out["normalized_boxes"],
            vltk.features: model_out["roi_features"],
        }


if __name__ == "__main__":
    # dataset = FRCNN.extract("/home/eltoto/vltk/tests/visualgenome")
    dataset = FRCNN.load("/home/eltoto/vltk/tests/visualgenome", split="train")
    entry = dataset.get(dataset.imgids[1])
    for k, v in entry.items():
        print(k, type(v))
    print(dataset)
