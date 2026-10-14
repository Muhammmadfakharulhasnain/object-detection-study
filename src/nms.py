"""Confidence filtering and Non-Maximum Suppression, written from scratch."""
import numpy as np
from .iou import calculate_iou, iou_matrix


def confidence_filter(boxes, scores, conf_threshold, classes=None):
    """Keep predictions with score >= conf_threshold. Returns filtered arrays (and classes)."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float)
    mask = scores >= conf_threshold
    if classes is None:
        return boxes[mask], scores[mask]
    return boxes[mask], scores[mask], np.asarray(classes)[mask]


def non_max_suppression_simple(boxes, scores, iou_threshold):
    """Readable reference version (pure Python loop over calculate_iou).

    Returns the indices of the kept boxes, highest score first.
    A box is suppressed when its IoU with an already kept box is > iou_threshold.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float)
    order = list(np.argsort(-scores, kind="stable"))
    keep = []
    while order:
        best = order.pop(0)
        keep.append(int(best))
        order = [i for i in order if calculate_iou(boxes[best], boxes[i]) <= iou_threshold]
    return keep


def non_max_suppression(boxes, scores, iou_threshold):
    """Same algorithm as the simple version, but computes IoU against all remaining
    boxes in one vectorised step. Single class. Returns kept indices, highest score first."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_matrix(boxes[best][None, :], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return keep


def class_aware_nms(boxes, scores, classes, iou_threshold):
    """Run NMS separately for every class, so a 'person' box can never suppress a 'chair' box.
    Returns kept indices sorted by descending score."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float)
    classes = np.asarray(classes)
    keep = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        kept = non_max_suppression(boxes[idx], scores[idx], iou_threshold)
        keep.extend(idx[kept].tolist())
    keep = np.array(keep, dtype=int)
    return keep[np.argsort(-scores[keep], kind="stable")] if len(keep) else keep
