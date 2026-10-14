"""Experiment helpers. Every function works on cached raw candidates (results/cache/*.pkl) and uses
only this repo's own NMS and evaluation code, so each number can be traced back to code in src/."""
import json
import os
import pickle
import numpy as np
import pandas as pd
from .nms import class_aware_nms
from .evaluation import Evaluator, apply_pipeline, SIZE_RANGES
from .analysis import analyse_errors

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "cache")
PERSON = 14


def load_gts(cache=CACHE):
    d = pickle.load(open(os.path.join(cache, "gts.pkl"), "rb"))
    return d["ids"], d["gts"]


def load_candidates(name, cache=CACHE):
    return pickle.load(open(os.path.join(cache, f"{name}.pkl"), "rb"))


def detections(cands, conf, nms_iou):
    """Confidence filter + class-aware NMS (our own implementation)."""
    return apply_pipeline(cands, conf, nms_iou, class_aware_nms)


def _row(m):
    return dict(n_det=m["n_det"], tp=m["tp"], fp=m["fp"], fn=m["fn"], precision=m["precision"],
                recall=m["recall"], f1=m["f1"])


def full_map(cands, gts, nms_iou=0.5, min_conf=0.01):
    """mAP@50 and mAP@50:95 from the full ranked list (confidence >= min_conf)."""
    ev = Evaluator(gts, detections(cands, min_conf, nms_iou))
    return ev.metrics(0.5)["map"], ev.map_50_95()


def conf_sweep(cands, gts, thresholds=(0.10, 0.25, 0.50, 0.75), nms_iou=0.5):
    rows = []
    for t in thresholds:
        ev = Evaluator(gts, detections(cands, t, nms_iou))
        m = ev.metrics(0.5)
        n_kept = sum(len(d["scores"]) for d in ev.dets.values())   # all kept detections, incl. ignored ones
        rows.append(dict(conf=t, **_row(m), dets_per_img=n_kept / len(gts), map50_truncated=m["map"]))
    return pd.DataFrame(rows)


def nms_sweep(cands, gts, nms_values=(0.3, 0.5, 0.7, None), conf=0.25):
    rows = []
    crowd = {i: g for i, g in gts.items() if int(((g["labels"] == PERSON) & ~g["difficult"]).sum()) >= 4}
    for t in nms_values:
        dets = detections(cands, conf, t)
        ev = Evaluator(gts, dets)
        m = ev.metrics(0.5)
        fps, _ = analyse_errors(gts, dets, cands, conf_thr=conf)
        ev_c = Evaluator(crowd, {i: dets[i] for i in crowd})
        mc = ev_c.metrics(0.5)
        full = Evaluator(gts, detections(cands, 0.01, t)).metrics(0.5)["map"]
        rows.append(dict(nms_iou=("off" if t is None else t), **_row(m),
                         dets_per_img=sum(len(d["scores"]) for d in dets.values()) / len(gts),
                         duplicate_fp=sum(f["kind"] == "duplicate" for f in fps),
                         crowded_person_recall=mc["tp_c"][PERSON] / max(mc["n_gt_c"][PERSON], 1),
                         crowded_person_gt=int(mc["n_gt_c"][PERSON]), n_crowded_images=len(crowd),
                         map50_conf0_01=full))
    return pd.DataFrame(rows)


def size_study(cands, gts, conf=0.25, nms_iou=0.5):
    ev = Evaluator(gts, detections(cands, conf, nms_iou))
    ev_full = Evaluator(gts, detections(cands, 0.01, nms_iou))
    rows = []
    for name, rng in list(SIZE_RANGES.items()) + [("all", None)]:
        m = ev.metrics(0.5, area_range=rng)
        rows.append(dict(size=name, n_gt=m["n_gt"], precision=m["precision"], recall=m["recall"],
                         ap50_at_conf0_01=ev_full.metrics(0.5, area_range=rng)["map"]))
    return pd.DataFrame(rows)


def condition_study(names, gts, conf=0.25, nms_iou=0.5):
    rows = []
    for n in names:
        c = load_candidates(n)
        dets = detections(c, conf, nms_iou)
        m = Evaluator(gts, dets).metrics(0.5)
        kept = sum(len(d["scores"]) for d in dets.values())
        map50, map5095 = full_map(c, gts, nms_iou)
        rows.append(dict(condition=n, precision=m["precision"], recall=m["recall"], f1=m["f1"],
                         dets_per_img=kept / len(gts), map50=map50, map50_95=map5095))
    df = pd.DataFrame(rows)
    base = df.iloc[0]
    for col in ("precision", "recall", "map50", "map50_95"):
        df[f"d_{col}"] = df[col] - base[col]
    return df


def timing():
    p = os.path.join(CACHE, "timing_batched.json")
    return json.load(open(p)) if os.path.exists(p) else {}
