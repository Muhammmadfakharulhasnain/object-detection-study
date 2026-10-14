"""Bounding-box utilities (NumPy only).

Two formats are used:
    xyxy   : (x_min, y_min, x_max, y_max)   -- corner format, used everywhere in this repo
    cxcywh : (x_center, y_center, width, height)

All functions accept a single box of shape (4,) or an array of boxes of shape (N, 4).
"""
import numpy as np


def xyxy_to_cxcywh(boxes):
    b = np.asarray(boxes, dtype=float)
    x1, y1, x2, y2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=-1)


def cxcywh_to_xyxy(boxes):
    b = np.asarray(boxes, dtype=float)
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def box_width(boxes):
    b = np.asarray(boxes, dtype=float)
    return np.maximum(0.0, b[..., 2] - b[..., 0])


def box_height(boxes):
    b = np.asarray(boxes, dtype=float)
    return np.maximum(0.0, b[..., 3] - b[..., 1])


def box_area(boxes):
    """Area of box(es). Degenerate boxes (x_max <= x_min or y_max <= y_min) have area 0."""
    return box_width(boxes) * box_height(boxes)


def intersection_box(box_a, box_b):
    """Rectangle shared by two boxes. May be degenerate if they do not overlap."""
    a, b = np.asarray(box_a, float), np.asarray(box_b, float)
    return np.array([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def intersection_area(box_a, box_b):
    return float(box_area(intersection_box(box_a, box_b)))


def union_area(box_a, box_b):
    return float(box_area(box_a) + box_area(box_b) - intersection_area(box_a, box_b))


def clip_boxes(boxes, width, height):
    """Clip boxes to the image rectangle [0, width] x [0, height]."""
    b = np.asarray(boxes, dtype=float).copy()
    b[..., [0, 2]] = np.clip(b[..., [0, 2]], 0, width)
    b[..., [1, 3]] = np.clip(b[..., [1, 3]], 0, height)
    return b
