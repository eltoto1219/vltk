import os
import unittest

from frcnn import Config, GeneralizedRCNN, Preprocess


PATH = os.path.dirname(os.path.realpath(__file__))
URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/images/input.jpg"


class TestFRCNNForwardPass(unittest.TestCase):
    def test_forward(self):

        # load models and model components
        frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        frcnn.roi_outputs.nms_thresh = [0.6]
        frcnn.roi_outputs.score_thresh = 0.2
        frcnn.roi_outputs.min_detections = 36
        frcnn.roi_outputs.max_detections = 36
        # Run the actual model
        image_preprocess = Preprocess(frcnn_cfg)
        images, sizes, scales_yx = image_preprocess(URL)
        output_dict = frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=frcnn_cfg.max_detections,
            return_tensors="np",
        )
