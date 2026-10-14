from common import *
c = [md('''# 06 - Threshold experiments

**Questions (guide, experiments 1 and 2):** how do the confidence threshold and the NMS IoU threshold change what the detector returns, and when does NMS become too aggressive?

Setup for every number below: YOLOv8n (COCO weights, no training), 640 px, 1,000 VOC2007-test images (seed 0), clean images, our own NMS and evaluator. Only one setting is changed at a time; the other is fixed (confidence study: NMS 0.5; NMS study: confidence 0.25).'''),
code(SETUP + '''
from src.experiments import *
from src.evaluation import Evaluator
from src.analysis import analyse_errors
ids, gts = load_gts(); cands = load_candidates("clean_640")'''),
md('''## Experiment 1 - confidence threshold

Kept detections = score >= threshold, then NMS (IoU 0.5). Precision / recall are computed at that operating point. `map50_truncated` is the AP of the **truncated** ranked list: a high threshold removes the low-confidence tail of the PR curve, so recall can never reach its full value and AP falls.'''),
code('''df1 = conf_sweep(cands, gts, (0.10, 0.25, 0.50, 0.75), nms_iou=0.5)
df1.to_csv(f"{RES}/confidence_threshold_results.csv", index=False); display(df1.round(4))'''),
code('''fine = np.round(np.arange(0.05, 0.96, 0.05), 2)
dff = conf_sweep(cands, gts, tuple(fine), 0.5)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(dff.conf, dff.precision, "o-", label="precision"); ax[0].plot(dff.conf, dff.recall, "s-", label="recall"); ax[0].plot(dff.conf, dff.f1, "^--", label="F1")
ax[0].axvline(0.25, c="gray", ls=":"); ax[0].set_xlabel("confidence threshold"); ax[0].legend(); ax[0].set_title("operating point")
ax[1].plot(dff.conf, dff.dets_per_img, "o-", c="C3"); ax[1].axhline(df1.attrs.get("g", 2.422), c="k", ls=":"); ax[1].set_xlabel("confidence threshold"); ax[1].set_ylabel("detections per image")
ax[1].axhline(np.mean([(~g["difficult"]).sum() for g in gts.values()]), c="k", ls=":", label="true objects per image"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{FIG}/confidence_threshold_sweep.png", dpi=110); plt.show()
dff.to_csv(f"{RES}/confidence_threshold_fine_sweep.csv", index=False)
best = dff.loc[dff.f1.idxmax()]; print(f"F1 is highest at confidence {best.conf:.2f} (F1 {best.f1:.3f}, precision {best.precision:.3f}, recall {best.recall:.3f})")'''),
md('''## Experiment 2 - NMS IoU threshold

Confidence fixed at 0.25. `duplicate_fp` counts false positives that sit on an object already found by a better box (what NMS is supposed to remove). `crowded_person_recall` is the recall for *people* in images that contain >= 4 people (where neighbouring boxes overlap most: aggressive NMS should hurt there). `map50_conf0_01` is mAP@50 with the full ranked list for each NMS setting.'''),
code('''df2 = nms_sweep(cands, gts, (0.3, 0.5, 0.7, None), conf=0.25)
df2.to_csv(f"{RES}/nms_threshold_results.csv", index=False); display(df2.round(4))'''),
code('''fine_n = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
dfn = nms_sweep(cands, gts, fine_n, conf=0.25)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ax[0].plot(dfn.nms_iou.astype(float), dfn.duplicate_fp, "o-"); ax[0].set_title("duplicate false positives"); 
ax[1].plot(dfn.nms_iou.astype(float), dfn.crowded_person_recall, "o-", c="C1"); ax[1].set_title(f"person recall, crowded images (n={int(dfn.crowded_person_gt[0])} people)")
ax[2].plot(dfn.nms_iou.astype(float), dfn.precision, "o-", label="precision"); ax[2].plot(dfn.nms_iou.astype(float), dfn.recall, "s-", label="recall"); ax[2].legend(); ax[2].set_title("all images")
for a in ax: a.set_xlabel("NMS IoU threshold")
plt.tight_layout(); plt.savefig(f"{FIG}/nms_threshold_sweep.png", dpi=110); plt.show(); dfn.to_csv(f"{RES}/nms_threshold_fine_sweep.csv", index=False)'''),
md("## Joint grid: confidence x NMS threshold\n\nThe two settings interact, so the full grid (F1 at the operating point; precision and recall in the CSV)."),
code('''rows = []
for cth in (0.10, 0.25, 0.50, 0.75):
    for nth in (0.3, 0.5, 0.7, None):
        m = Evaluator(gts, detections(cands, cth, nth)).metrics(0.5)
        rows.append(dict(conf=cth, nms_iou="off" if nth is None else nth, precision=m["precision"], recall=m["recall"], f1=m["f1"], n_det=m["n_det"]))
grid = pd.DataFrame(rows); grid.to_csv(f"{RES}/threshold_results.csv", index=False)
display(grid.pivot(index="conf", columns="nms_iou", values="f1").round(3)); display(grid.pivot(index="conf", columns="nms_iou", values="precision").round(3)); display(grid.pivot(index="conf", columns="nms_iou", values="recall").round(3))'''),
md("## Observation (numbers are produced by the cells above and printed here)"),
code('''o = df1.set_index("conf"); n = df2.set_index("nms_iou")
display(Markdown(f"""
**Confidence.** Raising the threshold from 0.10 to 0.75 cuts detections per image from {o.dets_per_img[0.10]:.2f} to {o.dets_per_img[0.75]:.2f}
(true objects per image: {np.mean([(~g['difficult']).sum() for g in gts.values()]):.2f}), precision goes {o.precision[0.10]:.3f} -> {o.precision[0.75]:.3f} and recall {o.recall[0.10]:.3f} -> {o.recall[0.75]:.3f}.
F1 peaks at {best.conf:.2f}. mAP@50 of the truncated list falls from {o.map50_truncated[0.10]:.3f} to {o.map50_truncated[0.75]:.3f}, which is why mAP should be computed with a very low threshold.

**NMS.** Without NMS there are {n.dets_per_img['off']:.1f} boxes per image and {int(n.duplicate_fp['off'])} duplicate false positives; with IoU 0.5 only {n.dets_per_img[0.5]:.2f} boxes and {int(n.duplicate_fp[0.5])} duplicates.
Moving 0.3 -> 0.7 changes duplicates from {int(n.duplicate_fp[0.3])} to {int(n.duplicate_fp[0.7])} and crowded-scene person recall from {n.crowded_person_recall[0.3]:.3f} to {n.crowded_person_recall[0.7]:.3f}
({int(n.crowded_person_gt[0.3])} people in {int(n.n_crowded_images[0.3])} crowded images, so that last comparison is based on a small sample).
"""))'''),
md('''Reading the output above: NMS is not free. A lower threshold removes more duplicates but starts deleting correct detections of touching objects (crowded images); a higher threshold keeps those objects but lets duplicates through. Because the two thresholds act on different things (confidence on individual scores, NMS on overlap between boxes) they have to be tuned together, as the grid shows.''')]
save(c, "06_threshold_experiments")
