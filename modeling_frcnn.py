import itertools
import math
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torchvision.ops import RoIPool
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms

from image_processor_frcnn import _clip_box


# Helper Functions
def build_backbone(cfg):
    input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    norm = cfg.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
        caffe_maxpool=cfg.MODEL.MAX_POOL,
    )
    freeze_at = cfg.BACKBONE.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        # may need to get back frozen batch norm
        # stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    # fmt: off
    out_features = cfg.RESNETS.OUT_FEATURES
    depth = cfg.RESNETS.DEPTH
    num_groups = cfg.RESNETS.NUM_GROUPS
    width_per_group = cfg.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels = cfg.RESNETS.STEM_OUT_CHANNELS
    out_channels = cfg.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1 = cfg.RESNETS.STRIDE_IN_1X1
    res5_dilation = cfg.RESNETS.RES5_DILATION
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[
        depth
    ]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [
        {"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features
    ]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "bottleneck_channels": bottleneck_channels,
            "out_channels": out_channels,
            "num_groups": num_groups,
            "norm": norm,
            "stride_in_1x1": stride_in_1x1,
            "dilation": dilation,
        }

        stage_kargs["block_class"] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)

    return ResNet(stem, stages, out_features=out_features)


def get_norm(norm, out_channels):
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
            "": lambda x: x,
        }[norm]
    return norm(out_channels)


def _create_grid_offsets(
    size: List[int], stride: int, offset: float, device: torch.device
):

    grid_height, grid_width = size
    shifts_x = torch.arange(
        offset * stride,
        grid_width * stride,
        step=stride,
        dtype=torch.float32,
        device=device,
    )
    shifts_y = torch.arange(
        offset * stride,
        grid_height * stride,
        step=stride,
        dtype=torch.float32,
        device=device,
    )

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


def find_top_rpn_proposals(
    proposals,
    pred_objectness_logits,
    images,
    image_sizes,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_side_len,
    training,
):
    """Args:
        proposals (list[Tensor]): (L, N, Hi*Wi*A, 4).
        pred_objectness_logits: tensors of lenngth L.
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): before nms
        post_nms_topk (int): after nms
        min_box_side_len (float): minimum proposal box side
        training (bool): True if proposals are to be used in training,
    Returns:
        proposals (list[Instances]): list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i.
    """
    num_images = len(images)
    device = proposals[0].device

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = torch.arange(num_images, device=device)
    for level_id, proposals_i, logits_i in zip(
        itertools.count(), proposals, pred_objectness_logits
    ):
        Hi_Wi_A = logits_i.shape[1]
        num_proposals_i = min(pre_nms_topk, Hi_Wi_A)

        # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
        # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_scores_i = logits_i[batch_idx, :num_proposals_i]
        topk_idx = idx[batch_idx, :num_proposals_i]

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(
            torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device)
        )

    # 2. Concat all levels together
    topk_scores = torch.cat(topk_scores, dim=1)
    topk_proposals = torch.cat(topk_proposals, dim=1)
    level_ids = torch.cat(level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results.
    results = []
    for n, image_size in enumerate(image_sizes):
        boxes = topk_proposals[n]
        scores_per_img = topk_scores[n]
        # I will have to take a look at the boxes clip method
        _clip_box(boxes, image_size)

        # filter empty boxes
        keep = _nonempty_boxes(boxes, threshold=min_box_side_len)
        lvl = level_ids
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = (
                boxes[keep],
                scores_per_img[keep],
                level_ids[keep],
            )

        keep = batched_nms(boxes, scores_per_img, lvl, nms_thresh)
        keep = keep[:post_nms_topk]

        res = {
            "size": image_size,
            "proposal_boxes": boxes[keep],
            "objectness_logits": scores_per_img[keep],
        }
        results.append(res)

    return results


def _nonempty_boxes(box, threshold: float = 0.0) -> torch.Tensor:
    """
    Find boxes that are non-empty.
    A box is considered empty, if either of its side is no larger than threshold.
    Returns:
        Tensor:
            a binary vector which represents whether each box is empty
            (False) or non-empty (True).
    """
    widths = box[:, 2] - box[:, 0]
    heights = box[:, 3] - box[:, 1]
    keep = (widths > threshold) & (heights > threshold)
    return keep


def batched_nms(
    boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for i in torch.jit.annotate(List[int], torch.unique(idxs).cpu().tolist()):
        mask = (idxs == i).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


def pairwise_intersection(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the intersection area between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax)
    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.
    Returns:
        Tensor: intersection, sized [N,M].
    """
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.
    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def subsample_labels(labels, num_samples, positive_fraction, bg_label):
    """
    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    positive = torch.nonzero((labels != -1) & (labels != bg_label)).squeeze(1)
    negative = torch.nonzero(labels == bg_label).squeeze(1)

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx


def rpn_losses(
    gt_objectness_logits,
    gt_anchor_deltas,
    pred_objectness_logits,
    pred_anchor_deltas,
    smooth_l1_beta,
):
    pass
    """
    see https://github.com/airsplay/py-bottom-up-attention/\
            blob/master/detectron2/modeling/roi_heads/roi_heads.py
    """


@torch.no_grad()
def frcnn_output(
        boxes,
        box_scores,
        attrs,
        attr_scores,
        feature_pooled,
        image_size,
        score_thresh,
        nms_thresh,
        topk_per_image,
):
    scores = box_scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = boxes.reshape(-1, 4)
    _clip_box(boxes, image_size[0])
    boxes = boxes.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = torch.nonzero(filter_mask, as_tuple=False)
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # Apply per-class NMS
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]

    boxes, scores, attrs, attr_scores, feature_pooled, filter_inds = (
        boxes[keep],
        scores[keep],
        attrs[keep],
        attr_scores[keep],
        filter_inds[keep],
        feature_pooled[keep]
    )
    return {
        "pred_boxes": boxes,
        "pred_obj_scores": scores,
        "pred_obj_classes": filter_inds[:, 1],
        "pred_attr_classes": attrs,
        "pred_attr_scores": attr_scores,
        "pred_features": feature_pooled
    }


def add_ground_truth_to_proposals(gt_boxes, proposals):
    assert gt_boxes is not None

    assert len(proposals) == len(gt_boxes)
    if len(proposals) == 0:
        return proposals

    return [
        add_ground_truth_to_proposals_single_image(gt_boxes_i, proposals_i)
        for gt_boxes_i, proposals_i in zip(gt_boxes, proposals)
    ]


def add_ground_truth_to_proposals_single_image(gt_boxes, proposals):
    """
    Augment `proposals` with ground-truth boxes from `gt_boxes`.
    Args:
        Same as `add_ground_truth_to_proposals`, but with gt_boxes and proposals
        per image.
    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    device = proposals.objectness_logits.device
    # Assign all ground-truth boxes an objectness logit corresponding to
    # P(object) = sigmoid(logit) =~ 1.
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)

    # Concatenating gt_boxes with proposals requires them to have the same fields
    gt_proposals = {
        "size": proposals["image_size"],
        "proposal_boxes": gt_boxes,
        "objectness_logits": gt_logits,
    }
    new_proposals = {**proposals, **gt_proposals}

    return new_proposals


def _fmt_box_list(box_tensor, batch_index: int):
    repeated_index = torch.full(
        (len(box_tensor), 1),
        batch_index,
        dtype=box_tensor.dtype,
        device=box_tensor.device,
    )
    return torch.cat((repeated_index, box_tensor), dim=1)


def convert_boxes_to_pooler_format(box_lists: List[torch.Tensor]):
    pooler_fmt_boxes = torch.cat(
        [_fmt_box_list(box_list, i) for i, box_list in enumerate(box_lists)],
        dim=0,
    )
    return pooler_fmt_boxes


def assign_boxes_to_levels(
    box_lists: List[torch.Tensor],
    min_level: int,
    max_level: int,
    canonical_box_size: int,
    canonical_level: int,
):

    box_sizes = torch.sqrt(torch.cat([boxes.area() for boxes in box_lists]))
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8)
    )
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level


# Helper Classes
class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    def __new__(cls, *, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)


class Box2BoxTransform(object):
    """
    This R-CNN transformation scales the box's width and height
    by exp(dw), exp(dh) and shifts a box's center by the offset
    (dx * width, dy * height).
    """

    def __init__(
        self, weights: Tuple[float, float, float, float], scale_clamp: float = None
    ):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally set
                such that the deltas have unit variance; now they are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        if scale_clamp is not None:
            self.scale_clamp = scale_clamp
        else:
            """
            Value for clamping large dw and dh predictions.
            The heuristic is that we clamp such that dw and dh are no larger
            than what would transform a 16px box into a 1000px box
            (based on a small anchor, 16px, and a typical image size, 1000px).
            """
            self.scale_clamp = math.log(1000.0 / 16)

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).
        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

        target_widths = target_boxes[:, 2] - target_boxes[:, 0]
        target_heights = target_boxes[:, 3] - target_boxes[:, 1]
        target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights

        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)

        deltas = torch.stack((dx, dy, dw, dh), dim=1)
        assert (
            (src_widths > 0).all().item()
        ), "Input boxes to Box2BoxTransform are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.
        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2
        return pred_boxes


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.
    The matching is determined by the MxN match_quality_matrix, that characterizes
    how well each (ground-truth, prediction)-pair match each other. For example,
    if the elements are boxes, this matrix may contain box intersection-over-union
    overlap values.
    The matcher returns (a) a vector of length N containing the index of the
    ground-truth element m in [0, M) that matches to prediction n in [0, N).
    (b) a vector of length N containing the labels for each prediction.
    """

    def __init__(
        self,
        thresholds: List[float],
        labels: List[int],
        allow_low_quality_matches: bool = False,
    ):
        """
        Args:
            thresholds (list): a list of thresholds used to stratify predictions
                into levels.
            labels (list): a list of values to label predictions belonging at
                each level. A label can be one of {-1, 0, 1} signifying
                {ignore, negative class, positive class}, respectively.
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions with maximum match quality lower than high_threshold.
                See set_low_quality_matches_ for more details.
            For example,
                thresholds = [0.3, 0.5]
                labels = [0, -1, 1]
                All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training.
                All predictions with 0.3 <= iou < 0.5 will be marked with -1 and
                thus will be ignored.
                All predictions with 0.5 <= iou will be marked with 1 and
                thus will be considered as true positives.
        """
        # Add -inf and +inf to first and last position in thresholds
        thresholds = thresholds[:]
        assert thresholds[0] > 0
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        # Currently torchscript does not support all + generator
        assert all(
            [low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:])]
        )
        assert all([label_i in [-1, 0, 1] for label_i in labels])
        assert len(labels) == len(thresholds) - 1
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero`
                for selecting indices in :meth:`set_low_quality_matches_`).
        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i]
                indicates whether a prediction is a true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), 0, dtype=torch.int64
            )
            # When no gt boxes exist, we define IOU = 0 and therefore set labels
            # to `self.labels[0]`, which usually defaults to background class 0
            # To choose to ignore instead,
            # can make labels=[-1,0,-1,1] + set appropriate thresholds
            default_match_labels = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8
            )
            return default_matches, default_match_labels

        assert torch.all(match_quality_matrix >= 0)

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)

        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

        for (l, low, high) in zip(
            self.labels, self.thresholds[:-1], self.thresholds[1:]
        ):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)

        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.
        This function implements the RPN assignment case (i)
        in Sec. 3.1.2 of Faster R-CNN.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find the highest quality match available, even if it is low, including ties.
        # Note that the matches qualities must be positive due to the use of
        # `torch.nonzero`.
        of_quality_inds = match_quality_matrix == highest_quality_foreach_gt[:, None]
        if of_quality_inds.dim() == 0:
            (_, pred_inds_with_highest_quality) = (
                of_quality_inds.unsqueeze(0).nonzero().unbind(1)
            )
        else:
            (_, pred_inds_with_highest_quality) = of_quality_inds.nonzero().unbind(1)
        # If an anchor was labeled positive only due to a low-quality match
        # with gt_A, but it has larger overlap with gt_B,
        # it's matched index will still be gt_B.
        # This follows the implementation in Detectron,
        # and is found to have no significant impact.
        match_labels[pred_inds_with_highest_quality] = 1


class RPNOutputs(object):
    def __init__(
        self,
        box2box_transform,
        anchor_matcher,
        batch_size_per_image,
        positive_fraction,
        images,
        pred_objectness_logits,
        pred_anchor_deltas,
        anchors,
        boundary_threshold=0,
        gt_boxes=None,
        smooth_l1_beta=0.0,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for
                anchor-proposal transformations.
            anchor_matcher (Matcher): :class:`Matcher` instance for matching anchors to
                ground-truth boxes; used to determine training labels.
            batch_size_per_image (int): number of proposals to sample when training
            positive_fraction (float): target fraction of sampled proposals
            that should be positive
            images (ImageList): :class:`ImageList` instance representing N input images
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for anchors.
            pred_anchor_deltas (list[Tensor]): A list of L elements.
            Element i is a tensor of shape
                (N, A*4, Hi, Wi) representing the predicted "deltas"
                used to transform anchors
                to proposals.
            anchors (list[list[Boxes]]): A list of N elements.
            Each element is a list of L
                Boxes. The Boxes at (n, l) stores the entire anchor array
                for feature map l in image
                n (i.e. the cell anchors repeated over all locations
                in feature map (n, l)).
            boundary_threshold (int): if >= 0, then anchors that extend beyond the image
                boundary by more than boundary_thresh are not used in training.
                Set to a very large
                number or < 0 to disable this behavior. Only needed in training.
            gt_boxes (list[Boxes], optional): A list of N elements.
            Element i a Boxes storing
                the ground-truth ("gt") boxes for image i.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pred_objectness_logits = pred_objectness_logits
        self.pred_anchor_deltas = pred_anchor_deltas

        self.anchors = anchors
        self.gt_boxes = gt_boxes
        self.num_feature_maps = len(pred_objectness_logits)
        self.num_images = len(images)
        self.boundary_threshold = boundary_threshold
        self.smooth_l1_beta = smooth_l1_beta

    def _get_ground_truth(self):
        """
        gt_objectness_logits: list of N tensors, whose length is
            total number of anchors in image i (i.e., len(anchors[i])). Label values are
            in {-1, 0, 1}, with meanings: -1 = ignore;
            0 = negative class; 1 = positive class.
        gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """
        gt_objectness_logits = []
        gt_anchor_deltas = []
        # Concatenate anchors from all feature maps into a single Boxes per image
        anchors = torch.stack(self.anchors, dim=0)
        for image_size_i, anchors_i, gt_boxes_i in zip(
            self.image_sizes, anchors, self.gt_boxes
        ):
            """
            image_size_i: (h, w) for the i-th image
            anchors_i: anchors for i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = pairwise_iou(gt_boxes_i, anchors_i)
            matched_idxs, gt_objectness_logits_i = self.anchor_matcher(
                match_quality_matrix
            )
            # Matching is memory-expensive, may result in CPU tensors.
            gt_objectness_logits_i = gt_objectness_logits_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.boundary_threshold >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality
                # that is turned off by default in Detectron2
                anchors_inside_image = anchors_i.inside_box(
                    image_size_i, self.boundary_threshold
                )
                gt_objectness_logits_i[~anchors_inside_image] = -1

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor
                # is labeled as background
                gt_anchor_deltas_i = torch.zeros_like(anchors_i.tensor)
            else:
                # TODO wasted computation for ignored boxes
                matched_gt_boxes = gt_boxes_i[matched_idxs]
                gt_anchor_deltas_i = self.box2box_transform.get_deltas(
                    anchors_i.tensor, matched_gt_boxes.tensor
                )

            gt_objectness_logits.append(gt_objectness_logits_i)
            gt_anchor_deltas.append(gt_anchor_deltas_i)

        return gt_objectness_logits, gt_anchor_deltas

    def predict_proposals(self):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.
        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B), where B is box dimension (4 or 5).
        """
        proposals = []
        # Transpose anchors from images-by-feature-maps
        # (N, L) to feature-maps-by-images (L, N)
        # anchors = list(zip(*self.anchors))
        # For each feature map
        # (num of anchors, num of images, box_dim)
        for anchor in zip(self.anchors, self.pred_anchor_deltas):
            anchors_i, pred_anchor_deltas_i = anchor
            # CHANGE: 0 to -1
            B = anchors_i.size(-1)
            N, _, Hi, Wi = pred_anchor_deltas_i.shape
            # Reshape: (N, A*B, Hi, Wi) ->
            # (N, A, B, Hi, Wi) ->
            # (N, Hi, Wi, A, B) ->
            # (N*Hi*Wi*A, B)
            pred_anchor_deltas_i = (
                pred_anchor_deltas_i.view(N, -1, B, Hi, Wi)
                .permute(0, 3, 4, 1, 2)
                .reshape(-1, B)
            )
            # Concatenate all anchors to shape (N*Hi*Wi*A, B)
            # type(anchors_i[0]) is Boxes (B = 4) or RotatedBoxes (B = 5)
            # anchors_i = (anchors_i[0],) + anchors_i
            proposals_i = self.box2box_transform.apply_deltas(
                pred_anchor_deltas_i, anchors_i
            )
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        proposals = torch.stack(proposals)
        return proposals

    def predict_objectness_logits(self):
        """
        Return objectness logits in the same format as the proposals returned by
        :meth:`predict_proposals`.
        Returns:
            pred_objectness_logits (list[Tensor]) -> (N, Hi*Wi*A).
        """
        pred_objectness_logits = [
            # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).reshape(self.num_images, -1)
            for score in self.pred_objectness_logits
        ]
        return pred_objectness_logits


# Main Classes
class Conv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """
        Args:
            norm(nn.Module, optional): a normalization layer
            activation(callable(Tensor) -> Tensor): a callable activation function
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0 and self.training:
            assert not isinstance(self.norm, torch.nn.SyncBatchNorm)
        if x.numel() == 0:
            assert not isinstance(self.norm, torch.nn.GroupNorm)
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:],
                    self.padding,
                    self.dilation,
                    self.kernel_size,
                    self.stride,
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num_levels = 2
        self.in_feature = "res5"
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        # origianlly initizlied to c2_xavier_fill

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class BasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, norm="BN", caffe_maxpool=False):
        """
        Args:
            norm(str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre - defined string
        """
        super().__init__()
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        self.caffe_maxpool = caffe_maxpool
        # use pad 1 instead of pad zero
        # check out fvcore for special weight init methods

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        if self.caffe_maxpool:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0, ceil_mode=True)
        else:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 4  # = stride 2 conv -> stride 2 max pool


class ResNetBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        Args:
             in_channels (int):
             out_channels (int):
             stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        # try without for now
        # FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self


class BottleneckBlock(ResNetBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
    ):
        """
        Args:
            stride_in_1x1 (bool): when stride==2, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        # deleted fvcore init weights method

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class Backbone(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network backbones.
    """

    def __init__(self):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()

    @abstractmethod
    def forward(self):
        """
        Subclasses must override this method, but adhere to the same return type.
        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
        """
        pass

    @property
    def size_divisibility(self):
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return 0

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    # the properties below are not used any more

    @property
    def out_features(self):
        """deprecated"""
        return self._out_features

    @property
    def out_feature_strides(self):
        """deprecated"""
        return {f: self._out_feature_strides[f] for f in self._out_features}

    @property
    def out_feature_channels(self):
        """deprecated"""
        return {f: self._out_feature_channels[f] for f in self._out_features}


class ResNet(Backbone):
    def __init__(self, stem, stages, num_classes=None, out_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in:
                    "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super(ResNet, self).__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, ResNetBlockBase), block
                curr_channels = block.out_channels
            stage = nn.Sequential(*blocks)
            name = "res" + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = blocks[-1].out_channels

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with std of 0.01."
            nn.init.normal_(self.linear.weight, stddev=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(
                ", ".join(children)
            )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        # so somehow the number of images gets squeezed here
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @staticmethod
    def make_stage(
        block_class,
        num_blocks,
        first_stride=None,
        *,
        in_channels,
        out_channels,
        **kwargs,
    ):
        """
        Usually, layers that produce the same feature map spatial size
        are defined as one "stage".
        Under such definition, stride_per_block[1:] should all be 1.
        """
        if first_stride is not None:
            assert "stride" not in kwargs and "stride_per_block" not in kwargs
            kwargs["stride_per_block"] = [first_stride] + [1] * (num_blocks - 1)
        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith("_per_block"):
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the "
                        f"same length as num_blocks={num_blocks}."
                    )
                    newk = k[: -len("_per_block")]
                    assert (
                        newk not in kwargs
                    ), f"Cannot call make_stage with both {k} and {newk}!"
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v

            blocks.append(
                block_class(
                    in_channels=in_channels, out_channels=out_channels, **curr_kwargs
                )
            )
            in_channels = out_channels

        return blocks


class ROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        pooler_type,
        canonical_box_size=224,
        canonical_level=4,
    ):
        """
        Args:

        output_size(int, tuple[int] or list[int]):
        output size of the pooled region,
        e.g., 14 x 14. If tuple or list is given, the length must be 2.
        scales(list[float]): The scale for each low - level pooling op relative to
        the input image. For a feature map with stride s relative to the input
        image, scale is defined as a 1 / s. The stride must be power of 2.
        When there are multiple scales, they must form a pyramid, i.e. they must be
        a monotically decreasing geometric sequence with a factor of 1 / 2.
        sampling_ratio(int): The `sampling_ratio` parameter for the ROIAlign op.
        pooler_type(string): Name of the type of
        pooling operation that should be applied.
        For instance, "ROIPool" or "ROIAlignV2".
        canonical_box_size(int): A canonical box size
        in pixels(sqrt(box area)). The default
        is heuristically defined as 224 pixels in the FPN paper(based on ImageNet
                                                                pre - training).
        canonical_level(int): The feature map level index
        from which a canonically - sized box
        should be placed. The default is defined as
        level 4 (stride=16) in the FPN paper,
        i.e., a box of size 224x224 will be placed on the feature with stride = 16.
        The box placement for all boxes will be determined from their sizes w.r.t
        canonical_box_size. For example, a box whose area is 4x that of a canonical box
        should be used to pool features from feature level ``canonical_level + 1``.
        Note that the actual input feature maps given to this module may not have
        sufficiently many levels for the input boxes. If the boxes are too large or too
        small for the input feature maps, the closest level will be used.
        """
        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size

        assert pooler_type == "ROIPool"
        self.level_poolers = nn.ModuleList(
            RoIPool(output_size, spatial_scale=scale) for scale in scales
        )

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -math.log2(scales[0])
        max_level = -math.log2(scales[-1])
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert (
            len(scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert 0 < self.min_level and self.min_level <= self.max_level
        if len(scales) > 1:
            # When there is only one feature map,
            # canonical_level is redundant and we should not
            # require it to be a sensible value. Therefore we skip this assertion
            assert (
                self.min_level <= canonical_level and canonical_level <= self.max_level
            )
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size

    def forward(self, x, proposals):
        """
        Args:
            x(list[Tensor]): A list of feature maps of
            NCHW shape, with scales matching those
                used to construct this module.
            box_lists(list[Boxes] | list[RotatedBoxes]):
                A list of N Boxes or N RotatedBoxes,
                where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of: class: `ROIPooler`.
        Returns:
            Tensor:
                A tensor of shape(M, C, output_size, output_size)
                where M is the total number of
                boxes aggregated over all N batch images and C
                is the number of channels in `x`.
        """
        x = [v for v in x.values()]
        boxes = proposals["proposal_boxes"]
        # objectness_logits = proposals["objectness_logits"]
        # inds = proposals["inds"]
        num_level_assignments = len(self.level_poolers)
        assert (
            len(x[0]) == num_level_assignments
        ), "unequal value, num_level_assignments={},\
                but x is list of {} Tensors".format(
            num_level_assignments, len(x[0])
        )

        assert len(boxes) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(boxes)
        )

        pooler_fmt_boxes = convert_boxes_to_pooler_format(boxes)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        level_assignments = assign_boxes_to_levels(
            boxes,
            self.min_level,
            self.max_level,
            self.canonical_box_size,
            self.canonical_level,
        )

        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size, output_size),
            dtype=dtype,
            device=device,
        )

        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            inds = torch.nonzero(level_assignments == level).squeeze(1)
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            output[inds] = pooler(x_level, pooler_fmt_boxes_level)

        return output


class Res5ROIHeads(nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.
    It contains logic of cropping the regions, extract per-region features
    (by the res-5 block in this case), and make per-region predictions.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        self.batch_size_per_image = cfg.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.RPN.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh = cfg.RPN.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh = cfg.RPN.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img = cfg.DETECTIONS_PER_IMAGE
        self.in_features = cfg.RPN.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.RPN.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt = cfg.RPN.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg = cfg.RPN.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta = cfg.RPN.ROI_BOX_HEAD.SMOOTH_L1_BETA
        self.stage_channel_factor = 2 ** 3  # res5 is 8x res2
        self.out_channels = cfg.RESNETS.RES2_OUT_CHANNELS * self.stage_channel_factor

        self.proposal_matcher = Matcher(
            cfg.RPN.ROI_HEADS.IOU_THRESHOLDS,
            cfg.RPN.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.RPN.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

        pooler_resolution = cfg.RPN.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type = cfg.RPN.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio = cfg.RPN.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        res5_halve = cfg.RPN.ROI_BOX_HEAD.RES5HALVE
        use_attr = cfg.RPN.ROI_BOX_HEAD.ATTR
        num_attrs = cfg.RPN.ROI_BOX_HEAD.NUM_ATTRS

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5 = self._build_res5_block(cfg)
        if not res5_halve:
            """
            Modifications for VG in RoI heads (modeling/roi_heads/roi_heads.py):
            1. Change the stride of conv1 and shortcut in Res5.Block1 from 2 to 1
            2. Modifying all conv2 with (padding: 1 --> 2) and (dilation: 1 --> 2)
            """
            self.res5[0].conv1.stride = (1, 1)
            self.res5[0].shortcut.stride = (1, 1)
            for i in range(3):
                self.res5[i].conv2.padding = (2, 2)
                self.res5[i].conv2.dilation = (2, 2)
        self.box_predictor = FastRCNNOutputLayers(
            self.out_channels,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
            use_attr=use_attr,
            num_attrs=num_attrs,
        )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = self.stage_channel_factor  # res5 is 8x res2
        num_groups = cfg.RESNETS.NUM_GROUPS
        width_per_group = cfg.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group * stage_channel_factor
        out_channels = self.out_channels
        stride_in_1x1 = cfg.RESNETS.STRIDE_IN_1X1
        norm = cfg.RESNETS.NORM

        blocks = ResNet.make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks)

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
            matched_labels (Tensor): a vector of length N, the matcher's label
            gt_classes (Tensor): a vector of length M.
        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: Each sample is labeled as either a category in [0, num_classes)
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals
            # (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_image,
            self.positive_sample_fraction,
            self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Augment proposals with ground-truth boxes.
        Adding the gt boxes to the set of proposals
        ensures that the second stage components will have some positive
        examples from the start of training. For RPN, this augmentation improves
        convergence and empirically improves box AP on COCO by about 0.5

        Returns:
            - proposal_boxes: the proposal boxes
            - gt_boxes: the ground-truth box that the proposal is assigned to
            - gt_classes
        """
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []
        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = targets_per_image.gt_boxes.tensor.new_zeros(
                    (len(sampled_idxs), 4)
                )

                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt

    def forward(self, features, proposals, targets=None):

        if self.training:
            proposal_boxes = self.label_and_sample_proposals(proposals, targets)
            del targets
        else:
            proposal_boxes = proposals

        # aha, so it is the features that are casusing the problem
        box_features = self._shared_roi_transform(features, proposal_boxes)
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

        box_predictions = self.box_predictor(feature_pooled)

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            box_predictions,
            proposals,
            self.smooth_l1_beta,
            feature_pooled
        )

        if self.training:
            raise NotImplementedError()
            """
            see https://github.com/airsplay/py-bottom-up-attention/\
                    blob/master/detectron2/modeling/roi_heads/roi_heads.py
            """
        else:
            pred_instances = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        sizes = cfg.ANCHOR_GENERATOR.SIZES
        aspect_ratios = cfg.ANCHOR_GENERATOR.ASPECT_RATIOS
        self.strides = [x.stride for x in input_shape]
        self.offset = cfg.ANCHOR_GENERATOR.OFFSET
        assert 0.0 <= self.offset < 1.0, self.offset

        # fmt: on
        """
        sizes (list[list[int]]): sizes[i] is the list of anchor sizes to use
            for the i-th feature map. If len(sizes) == 1, then the same list of
            anchor sizes, given by sizes[0], is used for all feature maps. Anchor
            sizes are given in absolute lengths in units of the input image;
            they do not dynamically scale if the input image size changes.
        aspect_ratios (list[list[float]]): aspect_ratios[i] is the list of
            anchor aspect ratios to use for the i-th feature map. If
            len(aspect_ratios) == 1, then the same list of anchor aspect ratios,
            given by aspect_ratios[0], is used for all feature maps.
        strides (list[int]): stride of each input feature.
        """

        self.num_features = len(self.strides)
        self.cell_anchors = nn.ParameterList(
            self._calculate_anchors(sizes, aspect_ratios)
        )
        # self.register_parameter("cell_anchors", cell_anchors)
        self._spacial_feat_dim = 4

    def _calculate_anchors(self, sizes, aspect_ratios):
        # If one size (or aspect ratio) is specified and there are multiple feature
        # maps, then we "broadcast" anchors of that single size (or aspect ratio)
        # over all feature maps.
        if len(sizes) == 1:
            sizes *= self.num_features
        if len(aspect_ratios) == 1:
            aspect_ratios *= self.num_features
        assert self.num_features == len(sizes)
        assert self.num_features == len(aspect_ratios)

        cell_anchors = [
            self.generate_cell_anchors(s, a).float()
            for s, a in zip(sizes, aspect_ratios)
        ]

        return cell_anchors

    @property
    def box_dim(self):
        return self._spacial_feat_dim

    @property
    def num_cell_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                (See also ANCHOR_GENERATOR.SIZES and
                ANCHOR_GENERATOR.ASPECT_RATIOS in config)
                In standard RPN models, `num_cell_anchors`
                on every feature map is the same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for (size, stride, base_anchors) in zip(
            grid_sizes, self.strides, self.cell_anchors
        ):
            shift_x, shift_y = _create_grid_offsets(
                size, stride, self.offset, base_anchors.device
            )
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def generate_cell_anchors(
        self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
    ):
        """
        anchors are continious geometric rectangles
        centered on one feature map point sample.
        We can later build the set of anchors
        for the entire feature map by tiling these tensors

        Args:
            sizes (tuple[float]): Absolute size of the anchors in the units of the input
                image (the scaled input received by the network)
                The absolute size is given as the side length of a box.
            aspect_ratios (tuple[float]]): Aspect ratios of the boxes computed as box
                height / width.
        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
        """

        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return nn.Parameter(torch.Tensor(anchors))

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps on which to generate anchors.
        Returns:
            list[list[Boxes]]: a list of #image elements.
            Each is a list of #feature level Boxes.
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)

        # anchors_in_image = []
        # for anchors_per_feature_map in anchors_over_all_feature_maps:
        #     anchors_in_image.append(anchors_per_feature_map)

        return torch.stack(anchors_over_all_feature_maps)


class StandardRPNHead(nn.Module):
    """
    RPN classification and regression heads. Uses a 3x3 conv to produce a shared
    hidden state from which one 1x1 conv predicts objectness logits for each anchor
    and a second 1x1 conv predicts bounding-box deltas specifying how to deform
    each anchor into an object proposal.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        anchor_generator = AnchorGenerator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_cell_anchors)) == 1
        ), "Each level must have the same number of cell anchors"
        num_cell_anchors = num_cell_anchors[0]

        if cfg.PROPOSAL_GENERATOR.HIDDEN_CHANNELS == -1:
            hid_channels = in_channels
        else:
            hid_channels = cfg.PROPOSAL_GENERATOR.HIDDEN_CHANNELS
            # Modifications for VG in RPN (modeling/proposal_generator/rpn.py)
            # Use hidden dim  instead fo the same dim as Res4 (in_channels)

        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(
            in_channels, hid_channels, kernel_size=3, stride=1, padding=1
        )
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(
            hid_channels, num_cell_anchors, kernel_size=1, stride=1
        )
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(
            hid_channels, num_cell_anchors * box_dim, kernel_size=1, stride=1
        )

        for layer in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas


class RPN(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.min_box_side_len = cfg.PROPOSAL_GENERATOR.MIN_SIZE
        self.in_features = cfg.RPN.IN_FEATURES
        self.nms_thresh = cfg.RPN.NMS_THRESH
        self.batch_size_per_image = cfg.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction = cfg.RPN.POSITIVE_FRACTION
        self.smooth_l1_beta = cfg.RPN.SMOOTH_L1_BETA
        self.loss_weight = cfg.RPN.LOSS_WEIGHT
        # fmt: on

        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            True: cfg.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.RPN.POST_NMS_TOPK_TEST,
        }
        self.boundary_threshold = cfg.RPN.BOUNDARY_THRESH

        self.anchor_generator = AnchorGenerator(
            cfg, [input_shape[f] for f in self.in_features]
        )
        self.box2box_transform = Box2BoxTransform(weights=cfg.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.RPN.IOU_THRESHOLDS,
            cfg.RPN.IOU_LABELS,
            allow_low_quality_matches=True,
        )
        self.rpn_head = StandardRPNHead(cfg, [input_shape[f] for f in self.in_features])

    def forward(self, images, image_shapes, features, gt_boxes=None):
        """
        Args:
            images (torch.Tensor): input images of length `N`
            features (dict[str: Tensor])
            gt_instances
        Returns:
            proposals: list[Instances] or None
            loss: dict[Tensor]
        """
        features = [features[f] for f in self.in_features]
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        anchors = self.anchor_generator(features)
        outputs = RPNOutputs(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            self.boundary_threshold,
            gt_boxes,
            self.smooth_l1_beta,
        )

        if self.training:
            losses = {k: v * self.loss_weight for k, v in outputs.losses().items()}
        else:
            losses = {}

        with torch.no_grad():
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.
            proposals = find_top_rpn_proposals(
                outputs.predict_proposals(),
                outputs.predict_objectness_logits(),
                images,
                image_shapes,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_side_len,
                self.training,
            )
            # For RPN-only models, the proposals are the final output
            # inds = torch.stack(
            #     [p["objectness_logits"].sort(descending=True)[1] for p in proposals]
            # )
            proposal_boxes = torch.stack(
                [p["proposal_boxes"] for p in proposals]
            )
            objectness_logits = torch.stack(
                [p["objectness_logits"] for p in proposals]
            )

            return {"proposal_boxes": proposal_boxes, 'objectness_logits':
                    objectness_logits, "sizes": image_shapes}, losses


class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        box_predictions,
        proposals,
        smooth_l1_beta,
        feature_pooled
    ):

        pred_class_logits, pred_attr_logits, pred_proposal_deltas = box_predictions
        self.feature_pooled = feature_pooled
        self.box2box_transform = box2box_transform
        # is this dynamic?
        self.num_preds_per_image = [len(p) for p in proposals["proposal_boxes"]]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.pred_attr_logits = pred_attr_logits
        self.proposals = proposals["proposal_boxes"]
        self.image_size = proposals["sizes"]
        assert (
            not self.proposals.requires_grad
        ), "Proposals should not require gradients!"

        # The following fields should exist only when training.
        if hasattr(proposals, "gt_boxes") and hasattr(proposals, "gt_boxes"):
            self.gt_boxes = proposals["gt_boxes"]
            self.gt_classs = proposals["gt_classes"]

    def _log_accuracy(self):
        raise NotImplementedError()
        """
        see:
        https://github.com/airsplay/py-bottom-up-attention/blob\
                /master/detectron2/modeling/roi_heads/fast_rcnn.py
        """

    def softmax_cross_entropy_loss(self):
        self._log_accuracy()
        return F.cross_entropy(
            self.pred_class_logits, self.gt_classes, reduction="mean"
        )

    def smooth_l1_loss(self):
        raise NotImplementedError()
        """
        see:
        https://github.com/airsplay/py-bottom-up-attention/blob\
                /master/detectron2/modeling/roi_heads/fast_rcnn.py
        """

        """
        Compute the smooth L1 loss for box regression.
        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        if cls_agnostic_bbox_reg:
            pass
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            # gt_class_cols = torch.arange(box_dim, device=device)
        else:
            pass
            # fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k
            # are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            # gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(
            #    box_dim, device=device
            # )

        loss_box_reg = torch.nn.SmoothL1Loss(reduction="sum")
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def losses(self):
        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
        }

    def predict_boxes(self):
        # N = self.proposals.size(0)
        num_pred = self.proposals.size(-2)
        B = self.proposals.size(-1)
        K = self.pred_proposal_deltas.size(-1) // B
        pred_proposal_deltas = self.pred_proposal_deltas.view(num_pred * K, B)
        proposals = self.proposals.expand(K, num_pred, B).reshape(-1, B)
        boxes = self.box2box_transform.apply_deltas(
            pred_proposal_deltas,
            proposals
        )
        return boxes.view(num_pred, K * B).split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    @torch.no_grad()
    def inference(self, score_thresh, nms_thresh, topk_per_image):
        boxes = self.predict_boxes()[0]
        box_scores = self.predict_probs()[0]
        attrs = self.pred_attr_logits[..., :-1].softmax(-1)
        attr_probs, attrs = attrs.max(-1)
        feature_pooled = self.feature_pooled
        image_sizes = self.image_size
        print(boxes.int(), boxes.shape)
        raise Exception

        return frcnn_output(
            boxes,
            box_scores,
            attrs,
            attr_probs,
            feature_pooled,
            image_sizes,
            score_thresh,
            nms_thresh,
            topk_per_image,
        ), {}


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(
        self,
        input_size,
        num_classes,
        cls_agnostic_bbox_reg,
        box_dim=4,
        use_attr=False,
        num_attrs=-1,
    ):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int)
            cls_agnostic_bbox_reg (bool)
            box_dim (int)
        """
        super(FastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # (do + 1 for background class)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        self.use_attr = use_attr
        if use_attr:
            '''
            pass
            Modifications for VG in RoI heads (modeling/roi_heads/fast_rcnn.py))\n"
            f"\tEmbedding: {num_classes + 1} --> {input_size // 8}"
            f"\tLinear: {input_size + input_size // 8} --> {input_size // 4}"
            f"\tLinear: {input_size // 4} --> {num_attrs + 1}
            '''
            self.cls_embedding = nn.Embedding(num_classes + 1, input_size // 8)
            self.fc_attr = nn.Linear(input_size + input_size // 8, input_size // 4)
            self.attr_score = nn.Linear(input_size // 4, num_attrs + 1)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for item in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(item.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        if self.use_attr:
            _, max_class = scores.max(-1)  # [b, c] --> [b]
            cls_emb = self.cls_embedding(max_class)  # [b] --> [b, 256]
            x = torch.cat([x, cls_emb], -1)  # [b, 2048] + [b, 256] --> [b, 2304]
            x = self.fc_attr(x)
            x = F.relu(x)
            attr_scores = self.attr_score(x)
            return scores, attr_scores, proposal_deltas
        else:
            return scores, proposal_deltas


class GeneralizedRCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = RPN(
            cfg, self.backbone.output_shape()
        )
        self.roi_heads = Res5ROIHeads(cfg, self.backbone.output_shape())
        self.input_format = cfg.INPUT.FORMAT
        self.to(self.device)

    def forward(self, images, image_shapes, gt_boxes=None, proposals=None):
        """Args:
            image: Tensor, image in (C, H, W) format.
            instances (optional): groundtruth :class:`Instances`
            proposals (optional): :class:`Instances`, precomputed proposals.
        """
        if self.training:
            raise NotImplementedError()
            # return self.training(images, image_sizes, gt_boxes)
            # see GeneralizedRCNN in Detectron2

        else:
            return self.inference(
                images=images,
                image_shapes=image_shapes,
                gt_boxes=gt_boxes,
                proposals=proposals
            )

    def training(self, bactched_inputs):
        pass

    def inference(
        self,
        images,
        image_shapes,
        gt_boxes=None,
        proposals=None,
    ):

        features = self.backbone(images)

        if gt_boxes is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(
                    images,
                    image_shapes,
                    features,
                    None
                )
            else:
                assert proposals is not None
            # proposals: (N, P, B)
            results, _ = self.roi_heads(features, proposals, None)
        else:
            results = self.roi_heads.forward_with_given_boxes(
                features, gt_boxes
            )

        return results
