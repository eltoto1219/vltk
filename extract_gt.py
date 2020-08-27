# coding=utf-8
import argparse
import base64
import csv
import json
import math
import os
import random
import sys
import time
from detectron2.utils.visualizer import Visualizer
import cv2
import numpy as np
import torch
import tqdm
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs,\
    fast_rcnn_inference_single_image
from detectron2.modeling.box_regression import Box2BoxTransform
# from detectron2.structures.boxes import Boxes
from detectron2.data import MetadataCatalog
from collections import defaultdict
import PIL.Image
from PIL import Image
from torch import nn
from torchvision.ops import nms
from detectron2.structures import Boxes, Instances


csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
D2_ROOT = os.path.dirname(os.path.dirname(detectron2.__file__))  # Root of detectron2
NUM_OBJECTS = 36


parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', default=1, type=int, help='batch_size')
parser.add_argument('--model', default='res4', type=str, help='options:'
                    '"res4", "res5"; features come from)')
parser.add_argument('--test', default=False, type=bool)
parser.add_argument('--weight', default='vg', type=str)
parser.add_argument('--scaling', default=False, type=bool)
parser.add_argument('--indir', default='{}'.format(os.pardir), type=str)
parser.add_argument('--outfile', default="feats.csv", type=str)
parser.add_argument('--use_attrs', default=False, type=bool)
parser.add_argument('--boxes_file', default=None, type=str)
parser.add_argument('--test_name', default="img_test", type=str)
args = parser.parse_args()

TEST_NAME = args.test_name
ATTR = args.use_attrs
BOX_FILE = args.boxes_file
BOXES = True if BOX_FILE is not None else False


def makeoutfile(x):
    if x.count("/") != 0:
        path = x.split("/").pop(-1)
        os.mkdirs(path)
    try:
        os.mknod(x)
    except Exception:
        pass


def doit(detector, raw_image, raw_box, scaling):
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        if not scaling and BOXES:
            temp = []
            for b in raw_box:
                temp.append([
                    float(b[0]) * raw_width,
                    float(b[1]) * raw_height,
                    float(b[2]) * raw_width,
                    float(b[3]) * raw_height])
            raw_box = temp

        if BOXES:
            raw_box = Boxes(torch.Tensor(raw_box).cuda())
        image = detector.transform_gen.get_transform(raw_image).apply_image(raw_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        new_height, new_width = image.shape[:2]
        scale_x = 1. * new_width / raw_width
        scale_y = 1. * new_height / raw_height

        inputs = [{
            "image": image,
            "height": raw_height,
            "width": raw_width,
        }]

        images = detector.model.preprocess_image(inputs)
        # Run Backbone Res1-Res4
        features = detector.model.backbone(images.tensor)

        if BOXES:
            boxes = raw_box.clone()
            boxes.scale(scale_x=scale_x, scale_y=scale_y)
            proposal_boxes = [boxes]
        else:
            # Generate proposals with RPN
            proposals, _ = detector.model.proposal_generator(images, features, None)
            proposal = proposals[0]
            print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)
            # Run RoI head for each proposal (RoI Pooling + Res5)
            proposal_boxes = [x.proposal_boxes for x in proposals]

        # Run RoI head for each proposal (RoI Pooling + Res5)
        features = [features[f] for f in detector.model.roi_heads.in_features]
        box_features = detector.model.roi_heads._shared_roi_transform(
            features, proposal_boxes)
        feature_pooled = box_features.mean(dim=[2, 3])  # (sum_proposals, 2048), pooled to 1x1


        if BOXES:
            if ATTR:
                pred_class_logits, pred_attr_logits, pred_proposal_deltas =\
                    detector.model.roi_heads.box_predictor(feature_pooled)
                attr_prob = pred_attr_logits[..., :-1].softmax(-1)
                max_attr_prob, max_attr_label = attr_prob.max(-1)
                pred_class_prob = nn.functional.softmax(pred_class_logits, -1)
                pred_scores, pred_classes = pred_class_prob[..., :-1].max(-1)
                instances = Instances(
                    image_size=(raw_height, raw_width),
                    pred_boxes=raw_box,
                    scores=pred_scores,
                    pred_classes=pred_classes,
                    attr_scores=max_attr_prob,
                    attr_classes=max_attr_label
                )
            else:
                pred_class_logits, pred_proposal_deltas =\
                    detector.model.roi_heads.box_predictor(feature_pooled)
                pred_class_prob = nn.functional.softmax(pred_class_logits, -1)
                pred_scores, pred_classes = pred_class_prob[..., :-1].max(-1)
                instances = Instances(
                    image_size=(raw_height, raw_width),
                    pred_boxes=raw_box,
                    scores=pred_scores,
                    pred_classes=pred_classes,
                )
        else:
            if ATTR:
                pred_class_logits, pred_attr_logits, pred_proposal_deltas =\
                    detector.model.roi_heads.box_predictor(feature_pooled)
            else:
                (pred_class_logits, pred_proposal_deltas) =\
                    detector.model.roi_heads.box_predictor(feature_pooled)

            outputs = FastRCNNOutputs(
                detector.model.roi_heads.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                detector.model.roi_heads.smooth_l1_beta,
            )
            probs = outputs.predict_probs()[0]
            boxes = outputs.predict_boxes()[0]

            # NMS
            for nms_thresh in np.arange(0.5, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes, probs, image.shape[1:],
                    score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
                )
                if len(ids) == NUM_OBJECTS:
                    break

            instances = detector_postprocess(instances, raw_height, raw_width)
            feature_pooled = feature_pooled[ids].detach()

            if ATTR:
                max_attr_prob = max_attr_prob[ids].detach()
                max_attr_label = max_attr_label[ids].detach()
                instances.attr_scores = max_attr_prob
                instances.attr_classes = max_attr_label
            else:
                pass

        return instances, feature_pooled


def dump_features(writer, detector, pathXid, bboxes, scaling):
    img_paths, img_ids = zip(*pathXid)
    imgs = [cv2.imread(img_path) for img_path in img_paths]

    for img, img_id in zip(
            imgs, img_ids):
        if BOXES:
            bboxes = bboxes[img_id]
        else:
            bboxes = None
        instances, features = doit(detector, img , None, scaling)
        instances = instances.to('cpu')
        features = features.to('cpu')
        num_objects = len(instances)

        visualize(instances, img)
        print("Classes", instances.pred_classes)
        raise Exception

        item = {
            "img_id": img_id,
            "img_h": img.shape[0],
            "img_w": img.shape[1],
            "objects_id": base64.b64encode(instances.pred_classes.numpy()).decode(),  # int64
            "objects_conf": base64.b64encode(instances.scores.numpy()).decode(),  # float32
            "attrs_id": base64.b64encode(np.zeros(num_objects, np.int64)).decode(),  # int64
            "attrs_conf": base64.b64encode(np.zeros(num_objects, np.float32)).decode(),  # float32
            "num_boxes": num_objects,
            "boxes": base64.b64encode(instances.pred_boxes.tensor.numpy()).decode(),  # float32
            "features": base64.b64encode(features.numpy()).decode()  # float32
        }
        writer.writerow(item)


def extract_feat(outfile, detector, pathXid, bboxes, scaling):
    # Check existing images in tsv file.
    wanted_ids = set([image_id[1] for image_id in pathXid])
    found_ids = set()
    if os.path.exists(outfile):
        with open(outfile, 'r') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                found_ids.add(item['img_id'])
    missing = wanted_ids - found_ids

    # Extract features for missing images.
    missing_pathXid = list(filter(lambda x: x[1] in missing, pathXid))
    with open(outfile, 'a') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
        for start in tqdm.tqdm(range(0, len(pathXid), args.batchsize)):
            pathXid_trunk = missing_pathXid[start: start + args.batchsize]
            dump_features(writer, detector, pathXid_trunk, bboxes, scaling)


def load_image_ids(img_root):
    """images in the same directory are in the same split"""
    pathXid = []
    for name in os.listdir(img_root):
        idx = name.split(".")[0]
        pathXid.append((
            os.path.join(img_root, name),
            idx
        ))
    return pathXid


def build_model():
    # Build model and load weights.
    if args.weight == 'mask':
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(
            D2_ROOT, "configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        print("Load the Mask RCNN weight for ResNet101, pretrained on MS COCO segmentation. ")
        cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/model_final_a2914c.pkl"
    elif args.weight == 'obj':
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(
            D2_ROOT, "configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl"
    elif args.weight == 'vg':
        cfg = get_cfg()  # Renew the cfg file
        if ATTR:
            cfg.merge_from_file("configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
        else:
            cfg.merge_from_file("configs/VG-Detection/faster_rcnn_R_101_C4_caffemaxpool.yaml")
        # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
        if not ATTR:
            cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
        else:
            cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"
    elif args.weight == "oi":
        assert False, "no this weight"
    detector = DefaultPredictor(cfg)
    return detector


def visualize(instances, im):
    # print(im)
    # im = cv2.imread(im)
    # im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pred = instances.to('cpu')
    v = Visualizer(im[:, :, :], MetadataCatalog.get("vg"), scale=1.2)
    v = v.draw_instance_predictions(pred)
    showarray(v.get_image()[:, :, ::-1])


def showarray(a):
    a = np.uint8(np.clip(a, 0, 255))
    img = Image.fromarray(a)
    img.save("{}.jpg".format(TEST_NAME))


if __name__ == "__main__":

    makeoutfile(args.outfile)

    print(f"outfile:{args.outfile}")
    print(f"outdir:{args.indir}")

    print("loading vg objs + attrs")
    # load class stuff


    MetadataCatalog.get("vg").thing_classes = vg_classes

    print(MetadataCatalog.get("vg"))

    raise Exception


    if ATTR:
        print("including attrs")
        MetadataCatalog.get("vg").attr_classes = vg_attrs

    # load images
    print("loading imgs")
    pathXid = load_image_ids(args.indir)

    # laod bounding boxxes for open images
    print("loading boxes")
    if not BOXES:
        bboxes = None
    else:
        with open(args.boxes_file) as f:
            bboxes = json.load(f)

    print("loading model")
    detector = build_model()
    extract_feat(args.outfile, detector, pathXid, bboxes, args.scaling)
