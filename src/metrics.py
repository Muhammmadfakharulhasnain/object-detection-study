"""Detection metrics written from scratch: precision, recall, PR curve and Average Precision."""
import numpy as np


def precision_recall(tp, fp, fn):
    """Precision = TP / (TP + FP), Recall = TP / (TP + FN). Returns 0.0 when the denominator is 0."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def pr_curve(is_tp, n_gt):
    """Precision-recall curve for detections that are ALREADY sorted by descending confidence.

    is_tp : 1/0 array (1 = true positive, 0 = false positive), one entry per detection.
    n_gt  : number of ground-truth objects (the denominator of recall).
    Returns (recall, precision) arrays, one point per detection.
    """
    is_tp = np.asarray(is_tp, dtype=float)
    tp_cum = np.cumsum(is_tp)
    fp_cum = np.cumsum(1.0 - is_tp)
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return recall, precision


def average_precision(recall, precision, method="all_points"):
    """Area under the precision-recall curve.

    all_points : Pascal VOC 2010+ / COCO style. Make precision monotonically decreasing
                 (precision at recall r = max precision at any recall >= r), then sum the area.
    11_point   : Pascal VOC 2007 style. Mean of the interpolated precision at recall 0, 0.1, ..., 1.0.
    """
    recall = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)
    if recall.size == 0:
        return 0.0
    if method == "11_point":
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = precision[recall >= t].max() if np.any(recall >= t) else 0.0
            ap += p / 11.0
        return float(ap)

    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):          # monotone envelope, right to left
        mpre[i] = max(mpre[i], mpre[i + 1])
    changed = np.where(mrec[1:] != mrec[:-1])[0]    # points where recall changes
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))
