"""Detection evaluation (Pascal VOC protocol), written from scratch.

Data layout used everywhere:
    gts[image_id]  = {"boxes": (n,4) xyxy, "labels": (n,) int, "difficult": (n,) bool, ...}
    dets[image_id] = {"boxes": (m,4) xyxy, "scores": (m,),   "labels": (m,) int}
Detections must already have gone through NMS.

Matching rule (VOC): detections are visited in descending score order. A detection is compared
with every ground-truth box of the same class in the same image and takes the one with the
highest IoU.
    IoU >= threshold and that GT is 'ignored' (difficult / outside size range)  -> detection ignored
    IoU >= threshold and GT not matched yet                                      -> true positive
    IoU >= threshold and GT already matched                                      -> false positive (duplicate)
    IoU <  threshold                                                             -> false positive
Ground-truth boxes never matched by a true positive are false negatives.
"""
import numpy as np
from .boxes import box_area
from .iou import iou_matrix
from .metrics import average_precision, pr_curve, precision_recall, f1_score

IOU_RANGE = np.round(np.arange(0.5, 0.96, 0.05), 2)           # 0.50, 0.55, ..., 0.95
SIZE_RANGES = {"small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, np.inf)}


class Evaluator:
    def __init__(self, gts, dets, num_classes=20):
        self.gts, self.dets, self.num_classes = gts, dets, num_classes
        self._pairs = []          # one entry per (image, class) that has detections or ground truth
        for img, g in gts.items():
            d = dets.get(img, {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, int)})
            for c in np.union1d(np.unique(g["labels"]), np.unique(d["labels"])).astype(int):
                gi = np.where(g["labels"] == c)[0]
                di = np.where(d["labels"] == c)[0]
                di = di[np.argsort(-d["scores"][di], kind="stable")]
                ious = iou_matrix(d["boxes"][di], g["boxes"][gi])
                self._pairs.append(dict(img=img, cls=int(c), gi=gi, di=di, ious=ious))
        self._cache = {}

    # ------------------------------------------------------------------ matching
    def match(self, iou_thr=0.5, area_range=None):
        """Returns per-class dict with score, status (1=TP, 0=FP, -1=ignored), and n_gt."""
        key = (round(float(iou_thr), 4), area_range)
        if key in self._cache:
            return self._cache[key]
        scores = [[] for _ in range(self.num_classes)]
        status = [[] for _ in range(self.num_classes)]
        n_gt = np.zeros(self.num_classes, dtype=int)
        for p in self._pairs:
            g, d = self.gts[p["img"]], self.dets.get(p["img"])
            gi, di, ious = p["gi"], p["di"], p["ious"]
            ignore = g["difficult"][gi].copy() if len(gi) else np.zeros(0, bool)
            if area_range is not None and len(gi):
                a = box_area(g["boxes"][gi])
                ignore |= ~((a >= area_range[0]) & (a < area_range[1]))
            n_gt[p["cls"]] += int((~ignore).sum())
            matched = np.zeros(len(gi), dtype=bool)
            det_area = box_area(d["boxes"][di]) if (d is not None and len(di)) else np.zeros(0)
            for k in range(len(di)):
                s = d["scores"][di[k]]
                st = 0
                if len(gi):
                    j = int(np.argmax(ious[k]))
                    if ious[k, j] >= iou_thr:
                        if ignore[j]:
                            st = -1
                        elif not matched[j]:
                            matched[j] = True
                            st = 1
                    # unmatched detection outside the size range is not penalised
                if st == 0 and area_range is not None:
                    if not (area_range[0] <= det_area[k] < area_range[1]):
                        st = -1
                scores[p["cls"]].append(s)
                status[p["cls"]].append(st)
        out = {"scores": [np.array(s) for s in scores], "status": [np.array(s, dtype=int) for s in status], "n_gt": n_gt}
        self._cache[key] = out
        return out

    # ------------------------------------------------------------------ metrics
    def metrics(self, iou_thr=0.5, conf_thr=0.0, area_range=None, ap_method="all_points"):
        """Per-class AP plus operating-point precision/recall at conf_thr.

        AP is computed from the detections that survive conf_thr (so a high threshold truncates
        the PR curve and lowers AP). Operating-point precision/recall are micro-averaged:
        TP, FP, FN are summed over all classes first.
        """
        m = self.match(iou_thr, area_range)
        ap = np.full(self.num_classes, np.nan)
        tp_c = np.zeros(self.num_classes, int); fp_c = np.zeros(self.num_classes, int)
        for c in range(self.num_classes):
            s, st = m["scores"][c], m["status"][c]
            keep = (s >= conf_thr) & (st >= 0)
            s, st = s[keep], st[keep]
            order = np.argsort(-s, kind="stable")
            st = st[order]
            tp_c[c], fp_c[c] = int((st == 1).sum()), int((st == 0).sum())
            if m["n_gt"][c] > 0:
                rec, prec = pr_curve(st, m["n_gt"][c])
                ap[c] = average_precision(rec, prec, ap_method)
        fn_c = m["n_gt"] - tp_c
        tp, fp, fn = int(tp_c.sum()), int(fp_c.sum()), int(fn_c.sum())
        precision, recall = precision_recall(tp, fp, fn)
        return {"ap": ap, "map": float(np.nanmean(ap)), "tp": tp, "fp": fp, "fn": fn,
                "n_gt": int(m["n_gt"].sum()), "n_det": tp + fp, "precision": precision,
                "recall": recall, "f1": f1_score(precision, recall),
                "tp_c": tp_c, "fp_c": fp_c, "fn_c": fn_c, "n_gt_c": m["n_gt"]}

    def map_50_95(self, conf_thr=0.0, area_range=None):
        return float(np.mean([self.metrics(t, conf_thr, area_range)["map"] for t in IOU_RANGE]))

    def pr_curve_all(self, iou_thr=0.5, cls=None):
        """Precision-recall curve (all classes pooled, or one class)."""
        m = self.match(iou_thr)
        classes = range(self.num_classes) if cls is None else [cls]
        s = np.concatenate([m["scores"][c] for c in classes]); st = np.concatenate([m["status"][c] for c in classes])
        keep = st >= 0
        s, st = s[keep], st[keep]
        order = np.argsort(-s, kind="stable")
        n_gt = int(sum(m["n_gt"][c] for c in classes))
        rec, prec = pr_curve(st[order], n_gt)
        return rec, prec, s[order]

    def size_metrics(self, iou_thr=0.5, conf_thr=0.0):
        return {name: self.metrics(iou_thr, conf_thr, area_range=r) for name, r in SIZE_RANGES.items()}


def apply_pipeline(candidates, conf_thr, nms_iou, nms_fn, max_det=300):
    """Turn cached raw candidates into final detections: confidence filter, then NMS per class.

    candidates[image_id] = {"boxes","scores","labels"} (many overlapping boxes per object).
    nms_fn(boxes, scores, labels, iou_threshold) -> kept indices   (use src.nms.class_aware_nms)
    nms_iou=None disables NMS (every box that passes the confidence filter is kept).
    """
    out = {}
    for img, c in candidates.items():
        keep = c["scores"] >= conf_thr
        b, s, l = c["boxes"][keep], c["scores"][keep], c["labels"][keep]
        if nms_iou is not None and len(b):
            k = nms_fn(b, s, l, nms_iou)
        else:
            k = np.argsort(-s, kind="stable")
        k = np.asarray(k, dtype=int)[:max_det]
        out[img] = {"boxes": b[k], "scores": s[k], "labels": l[k]}
    return out
