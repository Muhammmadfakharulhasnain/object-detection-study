"""Independent checks of the from-scratch code.
1. Our NMS vs torchvision.ops.batched_nms on real detector output.
2. Our whole decode+filter+NMS pipeline vs Ultralytics' own predict() on identical input tensors.
3. Our evaluator vs pycocotools (COCO-style 101-point AP, 'difficult' boxes marked as crowd/ignored).
Writes results/cross_check.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np, torch, torchvision
from src.dataset import load_voc_subset, VOC_CLASSES, VOC_TO_COCO_IDX
from src.model import load_yolo, letterbox, to_tensor, forward_raw, decode_all80_argmax
from src.nms import class_aware_nms, non_max_suppression
from src.experiments import load_gts, load_candidates, detections
from src.evaluation import Evaluator

torch.set_num_threads(1)
voc = sys.argv[1]
res = {}
ids, gts = load_gts()
cands = load_candidates("clean_640")

# ---- 1. NMS vs torchvision on all 1000 images, several thresholds
for thr in (0.3, 0.5, 0.7):
    same = total = 0
    for i in ids:
        c = cands[i]; k = c["scores"] >= 0.25
        b, s, l = c["boxes"][k], c["scores"][k], c["labels"][k]
        if not len(b):
            same += 1; total += 1; continue
        mine = set(class_aware_nms(b, s, l, thr).tolist())
        tv = set(torchvision.ops.batched_nms(torch.tensor(b), torch.tensor(s), torch.tensor(l.astype(np.int64)), thr).tolist())
        same += (mine == tv); total += 1
    res[f"nms_vs_torchvision_iou{thr}"] = dict(identical_images=same, images=total)
    print("NMS", thr, same, "/", total, flush=True)

# ---- 2. pipeline vs ultralytics predict on identical tensors
net, yolo = load_yolo("yolov8n.pt")
_, _, paths = load_voc_subset(voc, 1000, 0)
n_ok = n_img = 0; max_diff = 0.0
for i in ids[:60]:
    im = cv2.imread(paths[i]); rgb, r, pad = letterbox(im, 640); x = to_tensor([rgb])
    pred = forward_raw(net, x)[0]
    b, s, c = decode_all80_argmax(pred, 0.25)
    b = np.clip(b, 0, 640)            # Ultralytics clips boxes to the network input frame
    keep = class_aware_nms(b, s, c, 0.5) if len(b) else np.zeros(0, int)
    mine = sorted([(int(c[k]), tuple(np.round(b[k], 1))) for k in keep])
    r_ = yolo.predict(x, conf=0.25, iou=0.5, imgsz=640, verbose=False, device="cpu")[0]
    ub = r_.boxes.xyxy.numpy(); uc = r_.boxes.cls.numpy().astype(int)
    theirs = sorted([(int(uc[k]), tuple(np.round(ub[k], 1))) for k in range(len(uc))])
    n_img += 1
    same_cls = [m[0] for m in mine] == [t[0] for t in theirs]
    if same_cls and len(mine) == len(theirs):
        d = max([np.abs(np.array(m[1]) - np.array(t[1])).max() for m, t in zip(mine, theirs)] + [0])
        max_diff = max(max_diff, float(d)); n_ok += d < 0.6
res["ultralytics_predict_equivalence"] = dict(images=n_img, identical_detections=int(n_ok), max_box_diff_px=max_diff)
print("ultralytics", res["ultralytics_predict_equivalence"], flush=True)

# ---- 3. evaluator vs pycocotools
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import io, contextlib
    dets = detections(cands, 0.01, 0.5)
    images, anns, k = [], [], 1
    for n, i in enumerate(ids):
        g = gts[i]; images.append(dict(id=n, width=g["width"], height=g["height"]))
        for b, l, d in zip(g["boxes"], g["labels"], g["difficult"]):
            w, h = b[2] - b[0], b[3] - b[1]
            anns.append(dict(id=k, image_id=n, category_id=int(l), bbox=[float(b[0]), float(b[1]), float(w), float(h)],
                             area=float(w * h), iscrowd=int(d))); k += 1
    gt = COCO(); gt.dataset = dict(images=images, annotations=anns, categories=[dict(id=c) for c in range(20)])
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    dl = []
    for n, i in enumerate(ids):
        d = dets[i]
        for b, s, l in zip(d["boxes"], d["scores"], d["labels"]):
            dl.append(dict(image_id=n, category_id=int(l), bbox=[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], score=float(s)))
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dl); E = COCOeval(gt, dt, "bbox"); E.evaluate(); E.accumulate(); E.summarize()
    ours = Evaluator(gts, dets)
    res["pycocotools"] = dict(coco_map50=float(E.stats[1]), coco_map50_95=float(E.stats[0]),
                              ours_map50=ours.metrics(0.5)["map"], ours_map50_95=ours.map_50_95())
    print("coco", res["pycocotools"], flush=True)
except Exception as e:
    res["pycocotools_error"] = repr(e); print("pycocotools failed", e)

json.dump(res, open("results/cross_check.json", "w"), indent=1)
