"""Error analysis: sort every false positive and false negative into a cause.

Operating point: detections with score >= conf_thr after NMS, matched to ground truth at iou_thr.
Overlap of 0.1 is used as the line between 'poorly localised' and 'unrelated background'.

False positive types
    duplicate        same class, IoU >= iou_thr with a GT that another detection already matched
    classification   IoU >= iou_thr with a GT of a DIFFERENT class
    localization     same class, 0.1 <= IoU < iou_thr with a GT
    background       overlaps no GT of any class by >= 0.1
False negative (missed GT) causes
    classification   a detection of another class covers it (IoU >= iou_thr)
    localization     a same-class detection overlaps it with 0.1 <= IoU < iou_thr
    nms_suppressed   a same-class candidate with IoU >= iou_thr and score >= conf_thr existed before NMS,
                     but NMS removed it
    low_confidence   a same-class candidate with IoU >= iou_thr existed, but its score < conf_thr
    missed           nothing useful overlaps it
"""
import numpy as np
from .boxes import box_area
from .iou import iou_matrix

FP_TYPES = ["duplicate", "classification", "localization", "background"]
FN_TYPES = ["classification", "localization", "nms_suppressed", "low_confidence", "missed"]


def size_bucket(area):
    return "small" if area < 32 ** 2 else ("medium" if area < 96 ** 2 else "large")


def analyse_errors(gts, dets, candidates, conf_thr=0.25, iou_thr=0.5, low_overlap=0.1):
    fps, fns = [], []
    for img, g in gts.items():
        d = dets[img]; cand = candidates[img]
        ious = iou_matrix(d["boxes"], g["boxes"])                        # (nd, ng)
        order = np.argsort(-d["scores"], kind="stable")
        matched = np.zeros(len(g["boxes"]), bool)
        det_status = np.zeros(len(order), int)                           # 1 TP, 0 FP, -1 ignored
        for k in order:
            same = np.where(g["labels"] == d["labels"][k])[0]
            st, j = 0, -1
            if len(same):
                j = same[np.argmax(ious[k, same])]
                if ious[k, j] >= iou_thr:
                    if g["difficult"][j]:
                        st = -1
                    elif not matched[j]:
                        matched[j] = True; st = 1
            det_status[k] = st
            if st == 0:
                best_same = ious[k, same].max() if len(same) else 0.0
                other = np.where(g["labels"] != d["labels"][k])[0]
                best_other = ious[k, other].max() if len(other) else 0.0
                if best_same >= iou_thr:
                    kind = "duplicate"
                elif best_other >= iou_thr:
                    kind = "classification"
                elif best_same >= low_overlap:
                    kind = "localization"
                else:
                    kind = "background"
                gt_cls = int(g["labels"][other[np.argmax(ious[k, other])]]) if (kind == "classification") else -1
                fps.append(dict(img=img, det=int(k), kind=kind, score=float(d["scores"][k]), cls=int(d["labels"][k]), gt_cls=gt_cls,
                                size=size_bucket(float(box_area(d["boxes"][k])))))
        # missed ground truth
        cand_keep = cand["scores"] >= 0.01
        for j in np.where(~matched & ~g["difficult"])[0]:
            gb, gl = g["boxes"][j], g["labels"][j]
            row_d = ious[:, j] if len(d["boxes"]) else np.zeros(0)
            other_cover = np.any((d["labels"] != gl) & (row_d >= iou_thr)) if len(row_d) else False
            same_d = (d["labels"] == gl)
            loc = np.any(same_d & (row_d >= low_overlap) & (row_d < iou_thr)) if len(row_d) else False
            ci = iou_matrix(cand["boxes"], gb[None, :])[:, 0] if len(cand["boxes"]) else np.zeros(0)
            same_c = cand["labels"] == gl
            nms_sup = np.any(same_c & (ci >= iou_thr) & (cand["scores"] >= conf_thr)) if len(ci) else False
            low = np.any(same_c & (ci >= iou_thr) & (cand["scores"] < conf_thr)) if len(ci) else False
            if other_cover: cause = "classification"
            elif loc: cause = "localization"
            elif nms_sup: cause = "nms_suppressed"
            elif low: cause = "low_confidence"
            else: cause = "missed"
            area = float(box_area(gb))
            fns.append(dict(img=img, gt=int(j), cause=cause, cls=int(gl), size=size_bucket(area),
                            truncated=bool(g.get("truncated", np.zeros(len(g["boxes"]), bool))[j]), area=area))
    return fps, fns
