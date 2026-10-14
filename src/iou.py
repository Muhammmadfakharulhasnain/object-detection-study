"""Intersection over Union, written from scratch.

calculate_iou : one pair of boxes, written step by step (easy to read and to test).
iou_matrix    : vectorised version, every box in A against every box in B.
"""
import numpy as np
from .boxes import box_area


def calculate_iou(box_a, box_b):
    """IoU of two boxes in (x_min, y_min, x_max, y_max) format."""
    # 1. corners of the intersection rectangle
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    # 2. intersection area (0 if the boxes do not overlap)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    # 3. union = area A + area B - intersection
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    # 4. guard against zero-area boxes
    if union <= 0:
        return 0.0
    return float(inter / union)


def iou_matrix(boxes_a, boxes_b):
    """Pairwise IoU. Returns an array of shape (len(boxes_a), len(boxes_b))."""
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
