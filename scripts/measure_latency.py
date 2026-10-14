"""Latency of the detector on THIS machine: batch size 1, 1 CPU thread, warm-up excluded.
Reports forward pass only, and the whole pipeline (preprocess + forward + decode + own NMS)."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np, torch
from src.dataset import load_voc_subset
from src.model import load_yolo, letterbox, to_tensor, forward_raw, decode_candidates
from src.nms import class_aware_nms

torch.set_num_threads(1)
voc = sys.argv[1]
ids, gts, paths = load_voc_subset(voc, 100, 1)
net, _ = load_yolo("yolov8n.pt")
out = {}
for size in (320, 480, 640, 800):
    fwd, tot = [], []
    for k, i in enumerate(ids):
        t0 = time.perf_counter()
        im = cv2.imread(paths[i])
        rgb, r, pad = letterbox(im, size)
        x = to_tensor([rgb])
        t1 = time.perf_counter()
        pred = forward_raw(net, x)
        t2 = time.perf_counter()
        c = decode_candidates(pred[0], r, pad, im.shape[:2])
        keep = c["scores"] >= 0.25
        class_aware_nms(c["boxes"][keep], c["scores"][keep], c["labels"][keep], 0.5)
        t3 = time.perf_counter()
        if k >= 10:                                   # first 10 images = warm-up
            fwd.append(1000 * (t2 - t1)); tot.append(1000 * (t3 - t0))
    out[size] = dict(forward_ms_mean=float(np.mean(fwd)), forward_ms_std=float(np.std(fwd)),
                     pipeline_ms_mean=float(np.mean(tot)), pipeline_ms_std=float(np.std(tot)), n=len(fwd))
    print(size, out[size], flush=True)
json.dump(out, open("results/latency.json", "w"), indent=1)
