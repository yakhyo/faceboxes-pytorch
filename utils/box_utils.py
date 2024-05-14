import numpy as np

import torch
from torch import nn, Tensor

from typing import Tuple


def xywh2xyxy(boxes: Tensor | np.ndarray) -> Tensor | np.ndarray:
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = boxes.clone() if isinstance(boxes, torch.Tensor) else np.copy(boxes)
    y[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # top left x
    y[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # top left y
    y[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # bottom right x
    y[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # bottom right y

    return y


def xyxy2xywh(boxes: Tensor | np.ndarray) -> Tensor | np.ndarray:
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right."""
    y = boxes.clone() if isinstance(boxes, torch.Tensor) else np.copy(boxes)
    y[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2  # x center
    y[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2  # y center
    y[..., 2] = boxes[..., 2] - boxes[..., 0]  # width
    y[..., 3] = boxes[..., 3] - boxes[..., 1]  # height

    return y


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
    """

    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    return inter, union


def jaccard(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """

    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou


def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def match(overlap_threshold, gt_boxes, prior_boxes, variances, gt_labels, loc_targets, conf_targets, batch_idx):
    """
    Matches each prior box with the ground truth box of the highest jaccard overlap,
    encodes the bounding boxes, and updates the location and confidence targets.

    Args:
        overlap_threshold (float): The overlap threshold used when matching boxes.
        gt_boxes (tensor): Ground truth boxes, Shape: [num_objects, num_priors].
        prior_boxes (tensor): Prior boxes from priorbox layers, shape: [num_priors, 4].
        variances (tensor): Variances corresponding to each prior coord, shape: [num_priors, 4].
        gt_labels (tensor): Class labels for the image, shape: [num_objects].
        loc_targets (tensor): Tensor to be filled with encoded location targets.
        conf_targets (tensor): Tensor to be filled with matched indices for confidence predictions.
        batch_idx (int): Current batch index.
    """
    # Compute jaccard overlap between ground truth boxes and prior boxes
    overlaps = jaccard(gt_boxes, xywh2xyxy(prior_boxes))
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # Ignore ground truths with low overlap
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    filtered_best_prior_idx = best_prior_idx[valid_gt_idx, :]
    if filtered_best_prior_idx.shape[0] <= 0:
        loc_targets[batch_idx] = 0
        conf_targets[batch_idx] = 0
        return

    # Find the best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    filtered_best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    # Ensure every ground truth matches with its prior of max overlap
    best_truth_overlap.index_fill_(0, filtered_best_prior_idx, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    matched_boxes = gt_boxes[best_truth_idx]  # Shape: [num_priors, 4]
    matched_labels = gt_labels[best_truth_idx]  # Shape: [num_priors]
    matched_labels[best_truth_overlap < overlap_threshold] = 0  # Label as background

    encoded_locs = encode(matched_boxes, prior_boxes, variances)
    loc_targets[batch_idx] = encoded_locs  # [num_priors, 4] encoded offsets to learn
    conf_targets[batch_idx] = matched_labels  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """
    Encode the coordinates of ground truth boxes based on jaccard overlap with the prior boxes.
    This encoded format is used during training to compare against the model's predictions.

    Args:
        matched (torch.Tensor): Ground truth coordinates for each prior in point-form, shape: [num_priors, 4].
        priors (torch.Tensor): Prior boxes in center-offset form, shape: [num_priors, 4].
        variances (list[float]): Variances of prior boxes

    Returns:
        torch.Tensor: Encoded boxes, Shape: [num_priors, 4]
    """

    # Calculate centers of ground truth boxes
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]

    # Normalize the centers with the size of the priors and variances
    g_cxcy /= (variances[0] * priors[:, 2:])

    # Calculate the sizes of the ground truth boxes
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]  # Use log to transform the scale

    # Concatenate normalized centers and sizes to get the encoded boxes
    encoded_boxes = torch.cat([g_cxcy, g_wh], dim=1)  # Concatenation along the last dimension

    return encoded_boxes


def decode(loc, priors, variances):
    """
    Decode locations from predictions using priors to undo
    the encoding done for offset regression at train time.

    Args:
        loc (tensor): Location predictions for loc layers, shape: [num_priors, 4]
        priors (tensor): Prior boxes in center-offset form, shape: [num_priors, 4]
        variances (list[float]): Variances of prior boxes

    Returns:
        tensor: Decoded bounding box predictions
    """
    # Compute centers of predicted boxes
    cxcy = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]

    # Compute widths and heights of predicted boxes
    wh = priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])

    # Convert center, size to corner coordinates
    boxes = torch.empty_like(loc)
    boxes[:, :2] = cxcy - wh / 2  # xmin, ymin
    boxes[:, 2:] = cxcy + wh / 2  # xmax, ymax

    return boxes


def log_sum_exp(x):
    """
    Utility function for computing log_sum_exp.
    This function is used to compute the log of the sum of exponentials of input elements.

    Args:
        x (torch.Tensor): conf_preds from conf layers

    Returns:
        torch.Tensor: The result of the log_sum_exp computation.
    """
    return torch.logsumexp(x, dim=1, keepdim=True)
