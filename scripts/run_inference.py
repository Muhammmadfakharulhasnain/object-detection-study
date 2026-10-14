"""Run YOLOv8n once per (condition, resolution) and cache raw candidate boxes.
All later experiments (thresholds, NMS, metrics) run offline on these caches with this repo's code.

usage: python scripts/run_inference.py --voc VOCdevkit/VOC2007 --n 1000
"""
import argparse, os, pickle, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.dataset import load_voc_subset
from src.model import load_yolo, run_candidates
from src.corruptions import CORRUPTIONS

p = argparse.ArgumentParser()
p.add_argument("--voc", required=True)
p.add_argument("--n", type=int, default=1000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--weights", default="yolov8n.pt")
p.add_argument("--out", default="results/cache")
p.add_argument("--only", default=None, help="comma separated run names to (re)compute")
a = p.parse_args()
torch.set_num_threads(os.cpu_count() or 1)
os.makedirs(a.out, exist_ok=True)

ids, gts, paths = load_voc_subset(a.voc, a.n, a.seed)
with open(os.path.join(a.out, "gts.pkl"), "wb") as f:
    pickle.dump({"ids": ids, "gts": gts, "seed": a.seed}, f)

net, _ = load_yolo(a.weights)
runs = {}
for size in (320, 480, 640, 800):                       # resolution study on clean images
    runs[f"clean_{size}"] = (size, CORRUPTIONS["clean"])
for name, fn in CORRUPTIONS.items():                    # corruption study at 640
    if name != "clean":
        runs[f"{name}_640"] = (640, fn)
if a.only:
    runs = {k: v for k, v in runs.items() if k in a.only.split(",")}

timing_path = os.path.join(a.out, "timing_batched.json")
timing = json.load(open(timing_path)) if os.path.exists(timing_path) else {}
for name, (size, fn) in runs.items():
    out_path = os.path.join(a.out, f"{name}.pkl")
    if os.path.exists(out_path):
        print("skip", name); continue
    t0 = time.time()
    cands, ms = run_candidates(net, paths, size, transform=fn)
    with open(out_path, "wb") as f:
        pickle.dump(cands, f)
    timing[name] = ms
    json.dump(timing, open(timing_path, "w"), indent=1)
    print(f"{name}: {len(cands)} images, {ms:.1f} ms/img forward, {time.time()-t0:.0f}s total, "
          f"{sum(len(c['scores']) for c in cands.values())/len(cands):.0f} candidates/img", flush=True)
