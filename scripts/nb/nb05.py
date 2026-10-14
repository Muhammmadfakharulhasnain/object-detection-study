from common import *
c = [md('''# 05 - YOLO baseline

**Question:** what does a modern one-stage detector output, and which parts of the pipeline are the framework's and which are ours?'''),
code(SETUP + '''
import cv2, torch
torch.set_num_threads(1)
from src.dataset import load_voc_subset, VOC_CLASSES, VOC_TO_COCO_IDX, VOC_TO_COCO_NAME
from src.model import load_yolo, letterbox, to_tensor, forward_raw, decode_candidates
from src.nms import class_aware_nms
from src.visualization import gt_vs_pred, draw_boxes, show_image
from src.experiments import load_gts, load_candidates, detections
from src.evaluation import Evaluator'''),
md('''## 1. Concept: one-stage detection

A two-stage detector first proposes regions and then classifies them. A one-stage detector such as YOLO predicts boxes and class scores **directly** for many image positions in a single forward pass:

```
image -> letterbox to 640x640 -> CNN backbone + neck + head -> one row per position:
         [x_center, y_center, width, height, score_class_1 ... score_class_80]
      -> (our code) confidence filter -> NMS -> final boxes
```

YOLOv8 (anchor-free) outputs 8,400 rows at 640 px. Nothing in that output removes duplicates, so NMS is a separate post-processing step - the step written from scratch in notebook 03.

**What is the framework's and what is ours**

| step | who |
|---|---|
| network weights, forward pass | Ultralytics `yolov8n.pt`, pretrained on COCO |
| letterbox resize, decode raw output, class mapping | `src/model.py` (ours) |
| confidence filter, NMS (per class) | `src/nms.py` (ours) |
| matching, precision/recall, AP/mAP | `src/evaluation.py`, `src/metrics.py` (ours) |

**Experimental setup**

* Detector: `yolov8n.pt`, **used as is** (no training, no fine-tuning).
* Data: 1,000 random images (seed 0) of Pascal VOC 2007 *test*; 20 classes.
* The model predicts 80 COCO classes. Each VOC class is mapped to its COCO twin (table below); other COCO classes are discarded.'''),
code('''net, yolo = load_yolo("../yolov8n.pt")
print("parameters: %.2f M" % (sum(p.numel() for p in net.parameters()) / 1e6))
display(pd.DataFrame({"VOC class": VOC_CLASSES, "COCO class": VOC_TO_COCO_NAME, "COCO index": VOC_TO_COCO_IDX}).T)'''),
md("## 2. One image through the pipeline"),
code('''ids, gts = load_gts(); _, _, paths = load_voc_subset(VOC, 1000, 0)
i = [k for k in ids if 3 <= (~gts[k]["difficult"]).sum() <= 5][1]
im = cv2.imread(paths[i]); rgb, r, pad = letterbox(im, 640)
print("original", im.shape[:2], "-> letterboxed", rgb.shape[:2], "scale %.3f, padding" % r, pad)
x = to_tensor([rgb]); raw = forward_raw(net, x)
print("raw output shape (batch, 4 + 80, positions):", raw.shape)
cand = decode_candidates(raw[0], r, pad, im.shape[:2], min_conf=0.01)
rows = []
for t in (0.01, 0.10, 0.25, 0.50, 0.75):
    k = cand["scores"] >= t; keep = class_aware_nms(cand["boxes"][k], cand["scores"][k], cand["labels"][k], 0.5)
    rows.append((t, int(k.sum()), len(keep)))
display(pd.DataFrame(rows, columns=["confidence threshold", "candidates kept", "boxes after NMS (IoU 0.5)"]))
print("ground truth objects in this image:", [VOC_CLASSES[l] for l in gts[i]["labels"]])'''),
md("## 3. Our pipeline == Ultralytics' own `predict()`\n\nSame input tensor, confidence 0.25, NMS 0.5. (Full check on 60 images is in `results/cross_check.json`, produced by `scripts/cross_check.py`.)"),
code('''from src.model import decode_all80_argmax
b, s, cl = decode_all80_argmax(raw[0], 0.25); keep = class_aware_nms(b, s, cl, 0.5)
mine = sorted((int(cl[k]), tuple(np.round(b[k], 1))) for k in keep)
res = yolo.predict(x, conf=0.25, iou=0.5, imgsz=640, verbose=False, device="cpu")[0]
theirs = sorted((int(c_), tuple(np.round(bb, 1))) for c_, bb in zip(res.boxes.cls.numpy().astype(int), res.boxes.xyxy.numpy()))
print("ours   :", len(mine), "boxes\\nultra. :", len(theirs), "boxes\\nsame classes and boxes within 0.1 px:",
      [m[0] for m in mine] == [t[0] for t in theirs] and all(np.abs(np.array(m[1]) - np.array(t[1])).max() < 0.15 for m, t in zip(mine, theirs)))
import json
if os.path.exists(f"{RES}/cross_check.json"):
    cc = json.load(open(f"{RES}/cross_check.json")); print({k: v for k, v in cc.items() if k != "pycocotools"})'''),
md("## 4. Qualitative baseline (confidence 0.25, NMS 0.5, 640 px)\n\nGreen solid = ground truth, red dashed = prediction with score."),
code('''cands = load_candidates("clean_640"); dets = detections(cands, 0.25, 0.5)
pick = [k for k in ids if 2 <= (~gts[k]["difficult"]).sum() <= 6][2:14:2]
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
for ax, k in zip(axes.ravel(), pick): gt_vs_pred(ax, paths[k], gts[k], dets[k], title=k)
plt.tight_layout(); plt.savefig(f"{FIG}/detection_examples.png", dpi=100); plt.show()'''),
md("## 5. Baseline numbers on the 1,000-image subset"),
code('''ev = Evaluator(gts, dets); m = ev.metrics(0.5)
ev_full = Evaluator(gts, detections(cands, 0.01, 0.5)); mf = ev_full.metrics(0.5)
base = pd.DataFrame([dict(model="YOLOv8n (COCO weights, zero-shot on VOC classes)", input=640, conf=0.25, nms_iou=0.5, images=len(ids), objects_evaluated=m["n_gt"],
    TP=m["tp"], FP=m["fp"], FN=m["fn"], precision=m["precision"], recall=m["recall"], f1=m["f1"], mAP50=mf["map"], mAP50_95=ev_full.map_50_95())])
base.to_csv(f"{RES}/baseline_results.csv", index=False); display(base.round(4).T)'''),
md('''## 6. Observation

* The raw network output is just 8,400 rows of boxes and scores; the list of detections a user sees only exists after a confidence filter and NMS. Reimplementing those two steps reproduces the framework's own output (section 3).
* Precision/recall in the table are an *operating point* (confidence 0.25, NMS 0.5); the mAP numbers use every prediction with confidence >= 0.01, which is the usual way to compute mAP.
* This is a **zero-shot evaluation of a COCO-trained model on VOC classes**, not a model trained for this project. VOC and COCO label conventions differ slightly (for example what counts as a `sofa` vs `couch`, or `diningtable`), which costs some AP that has nothing to do with detector quality.''')]
save(c, "05_yolo_baseline")
