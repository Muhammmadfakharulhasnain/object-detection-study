from common import *
c = [md('''# 03 - Non-Maximum Suppression from scratch

**Question:** a detector proposes many boxes for one object. How do we keep one per object, and how does the overlap threshold change the result?'''),
code(SETUP + '''
from src.iou import calculate_iou, iou_matrix
from src.nms import confidence_filter, non_max_suppression_simple, non_max_suppression, class_aware_nms
from src.visualization import draw_boxes, show_image
from src.dataset import VOC_CLASSES'''),
md('''## 1. Concept

A detector predicts at many grid positions, so neighbouring positions all fire on the same object. Two different filters clean this up and they are easy to mix up:

| filter | question it asks | parameter |
|---|---|---|
| confidence filter | is this single prediction confident enough? | confidence threshold |
| NMS | does this box overlap an already chosen, better box too much? | IoU threshold |

Algorithm (greedy):

```
sort boxes by score, highest first
while boxes remain:
    take the best box -> keep it
    remove every remaining box whose IoU with it is > iou_threshold
```

Direction to remember: **lower IoU threshold = more suppression** (boxes only need a little overlap to be removed); higher threshold = more duplicates survive.

NMS is run **per class**, otherwise a "person" box could delete an overlapping "chair" box.'''),
md("## 2. Worked example by hand\n\nThree predictions, as in the guide: scores 0.95, 0.91, 0.40. To make the threshold matter, B and C are copies of A shifted sideways by 30 and 50 pixels."),
code('''boxes = np.array([[0, 0, 100, 100], [30, 0, 130, 100], [50, 0, 150, 100.]])      # A, B, C
scores = np.array([0.95, 0.91, 0.40]); names = "ABC"
M = iou_matrix(boxes, boxes)
print(pd.DataFrame(M, index=list(names), columns=list(names)).round(3))
print("\\nA vs B = 7000/13000 = %.3f ; A vs C = 5000/15000 = %.3f ; B vs C = %.3f" % (M[0,1], M[0,2], M[1,2]))'''),
md("### Trace of the algorithm at IoU threshold 0.5"),
code('''order = list(np.argsort(-scores)); keep = []; log = []
while order:
    best = order.pop(0); keep.append(best); removed = []
    for j in list(order):
        iou = calculate_iou(boxes[best], boxes[j])
        if iou > 0.5: order.remove(j); removed.append(f"{names[j]} (IoU {iou:.2f} > 0.5)")
    log.append((names[best], "kept", ", ".join(removed) or "nothing removed", "".join(names[k] for k in order) or "-"))
display(pd.DataFrame(log, columns=["step: best box", "decision", "removed because of overlap", "still waiting"]))
print("final:", [names[k] for k in keep])'''),
md("### Three thresholds on the same boxes"),
code('''fig, axes = plt.subplots(1, 4, figsize=(18, 3.2)); rows = []
for ax, (title, thr) in zip(axes, [("before NMS", None), ("IoU thr 0.3", 0.3), ("IoU thr 0.5", 0.5), ("IoU thr 0.7", 0.7)]):
    kept = list(range(3)) if thr is None else non_max_suppression(boxes, scores, thr)
    ax.set_xlim(-10, 180); ax.set_ylim(140, -20); ax.set_aspect("equal")
    for k in range(3):
        draw_boxes(ax, [boxes[k]], color="green" if k in kept else "lightgray", lw=3 if k in kept else 1.5, linestyle="-" if k in kept else "--")
        ax.text(boxes[k][0] + 3, boxes[k][1] + 14 + 12 * k, f"{names[k]} {scores[k]:.2f}", fontsize=9)
    ax.set_title(title + ("" if thr is None else f"  -> keeps {''.join(names[k] for k in kept)}"))
    if thr: rows.append((thr, "".join(names[k] for k in kept)))
plt.tight_layout(); plt.savefig(f"{FIG}/nms_comparison.png", dpi=110); plt.show()
print(rows)
assert non_max_suppression(boxes, scores, 0.3) == [0] and non_max_suppression(boxes, scores, 0.5) == [0, 2] and non_max_suppression(boxes, scores, 0.7) == [0, 1, 2]'''),
md("## 3. Tests\n\nThe readable loop version and the vectorised version must give identical answers; class-aware NMS must not let one class suppress another."),
code('''import time
rng = np.random.RandomState(1)
xy = rng.rand(300, 2) * 100; wh = rng.rand(300, 2) * 40 + 5
B = np.concatenate([xy, xy + wh], 1); S = rng.rand(300)
for t in (0.3, 0.5, 0.7):
    assert non_max_suppression(B, S, t) == non_max_suppression_simple(B, S, t)
print("simple == vectorised for 300 random boxes at 0.3, 0.5, 0.7")
two = np.array([[0, 0, 10, 10], [0, 0, 10, 10.]])
assert len(class_aware_nms(two, [0.9, 0.8], [0, 1], 0.5)) == 2 and len(class_aware_nms(two, [0.9, 0.8], [0, 0], 0.5)) == 1
b, s = confidence_filter(boxes, scores, 0.5); assert len(b) == 2
print("class-aware and confidence filter tests passed")'''),
md("## 4. NMS on real detector output\n\nRaw YOLOv8n output at 640 px, confidence >= 0.25: the image with the most candidate boxes, before and after NMS."),
code('''from src.experiments import load_gts, load_candidates, detections
from src.dataset import load_voc_subset
ids, gts = load_gts(); cands = load_candidates("clean_640")
_, _, paths = load_voc_subset(VOC, 1000, 0)
n_cand = {i: int((cands[i]["scores"] >= 0.25).sum()) for i in ids}
i = max(n_cand, key=n_cand.get)
c = cands[i]; k = c["scores"] >= 0.25
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
show_image(axes[0], paths[i]); draw_boxes(axes[0], c["boxes"][k], c["labels"][k], c["scores"][k], color="red", lw=1)
axes[0].set_title(f"{i}: {k.sum()} candidates before NMS")
for ax, thr in zip(axes[1:], (0.3, 0.7)):
    d = detections({i: c}, 0.25, thr)[i]; show_image(ax, paths[i]); draw_boxes(ax, d["boxes"], d["labels"], d["scores"], color="lime", lw=2)
    ax.set_title(f"after NMS (IoU thr {thr}): {len(d['scores'])} boxes")
plt.tight_layout(); plt.savefig(f"{FIG}/nms_real_image.png", dpi=100); plt.show()'''),
md("### Over the whole subset: how many boxes survive?"),
code('''rows = []
for thr in (0.3, 0.5, 0.7, None):
    d = detections(cands, 0.25, thr)
    rows.append(dict(nms_iou="off" if thr is None else thr, boxes_per_image=np.mean([len(v["scores"]) for v in d.values()]),
                     gt_per_image=np.mean([(~g["difficult"]).sum() for g in gts.values()])))
df = pd.DataFrame(rows); display(df.round(3))'''),
md("### A crowded scene: NMS can remove correct neighbours\n\nImages with at least 4 people. Same detections, three NMS thresholds."),
code('''crowd = [i for i in ids if ((gts[i]["labels"] == 14) & ~gts[i]["difficult"]).sum() >= 4]
j = max(crowd, key=lambda x: (gts[x]["labels"] == 14).sum()); print(len(crowd), "images with >= 4 people; example", j, "has", (gts[j]["labels"] == 14).sum(), "people")
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, thr in zip(axes, (0.3, 0.5, 0.7)):
    d = detections({j: cands[j]}, 0.25, thr)[j]; p = d["labels"] == 14
    show_image(ax, paths[j]); draw_boxes(ax, gts[j]["boxes"][gts[j]["labels"] == 14], color="lime", lw=2)
    draw_boxes(ax, d["boxes"][p], color="red", lw=1.5, linestyle="--"); ax.set_title(f"NMS IoU {thr}: {p.sum()} person boxes (green = GT)")
plt.tight_layout(); plt.savefig(f"{FIG}/nms_crowded_scene.png", dpi=100); plt.show()'''),
md("### Class-aware vs class-agnostic NMS (whole subset)"),
code('''from src.evaluation import Evaluator, apply_pipeline
agn = lambda b, s, l, t: class_aware_nms(b, s, np.zeros(len(l), int), t)
rows = []
for name, fn in (("class-aware", class_aware_nms), ("class-agnostic", agn)):
    m = Evaluator(gts, apply_pipeline(cands, 0.25, 0.5, fn)).metrics(0.5)
    rows.append(dict(NMS=name, detections=m["n_det"], TP=m["tp"], FP=m["fp"], FN=m["fn"], precision=round(m["precision"], 4), recall=round(m["recall"], 4)))
display(pd.DataFrame(rows))'''),
md('''## 5. Observation

* In the hand example IoU(A,B) = 0.54 and IoU(A,C) = 0.33. Threshold 0.3 keeps only A, threshold 0.5 keeps A and C (B is removed), threshold 0.7 keeps all three. C's low score (0.40) plays no role in NMS; only a confidence filter would remove it.
* On a real image dozens of candidate boxes collapse to a handful (figure `nms_real_image.png`), and the table in section 4 shows how many boxes per image remain against the number of true objects per image.
* In the crowded scene a low threshold (0.3) merges neighbouring people into one box; a high threshold keeps more boxes but also leaves more duplicates. There is no universally right value, which is why this is tested as an experiment in notebook 06.
* Class-agnostic NMS lets a box of one class remove an overlapping box of another class; the last table shows what that costs on this data.''')]
save(c, "03_nms_from_scratch")
