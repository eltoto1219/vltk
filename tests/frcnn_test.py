import os
import unittest

import torch
from vltk import Config, GeneralizedRCNN, Preprocess, SingleImageViz, get_data

PATH = os.path.dirname(os.path.realpath(__file__))
URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/images/input.jpg"


class TestFRCNNForwardPass(unittest.TestCase):
    def test_forward(self):
        # load models and model components
        # frcnn_cfg = Config.from_pretrained(f"{PATH}/config.yaml")
        frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        frcnn.roi_outputs.nms_thresh = [0.5, 1.0, 0.1]
        frcnn.roi_outputs.score_thresh = 0.2
        frcnn.roi_outputs.min_detections = 36
        frcnn.roi_outputs.max_detections = 36
        frcnn_cfg = frcnn.config
        # Run the actual model
        image_preprocess = Preprocess(frcnn_cfg)
        images, sizes, scales_yx = image_preprocess(URL)
        frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=frcnn_cfg.max_detections,
            return_tensors="np",
        )


if __name__ == "__main__":
    Image = SingleImageViz(
        URL,
        id2obj=get_data("./tests/objects.txt"),
        id2attr=get_data("./tests/attributes.txt"),
    )

    # load models and model components
    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    frcnn.roi_outputs.min_detections = 36
    frcnn.roi_outputs.max_detections = 36
    # Run the actual model
    image_preprocess = Preprocess(frcnn.config)
    images, sizes, scales_yx = image_preprocess(URL)
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn.config.max_detections,
        return_tensors="np",
    )

    Image.draw_boxes(
        output_dict["boxes"],
        output_dict["obj_ids"],
        output_dict["obj_probs"],
        output_dict["attr_ids"],
        output_dict["attr_probs"],
    )
    Image.save('test.jpg')
