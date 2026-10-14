from common import *
c = [md('''# 07 - Object size, image conditions and input resolution

**Questions (guide, experiments 3, 4 and the resolution experiment):**
1. Is detection worse for small objects?
2. How much do blur, brightness, contrast, noise and JPEG compression hurt, and do the best confidence / NMS thresholds change under those conditions?
3. Does a larger input resolution buy enough accuracy to justify its cost?

Same detector (YOLOv8n, COCO weights), same 1,000 images, same ground truth, same evaluator in every row. Only the input changes. Corruptions do not change image geometry, so ground-truth boxes stay valid. Default operating point: confidence 0.25, NMS IoU 0.5. mAP columns use all predictions with confidence >= 0.01.'''),
code(SETUP + '''
import cv2, json
from src.experiments import *
from src.corruptions import CORRUPTIONS
from src.dataset import load_voc_subset
from src.evaluation import Evaluator
ids, gts = load_gts(); _, _, paths = load_voc_subset(VOC, 1000, 0)'''),
md('''## Experiment 3 - object size

Ground-truth objects are split by box area with the COCO limits: small < 32^2 px, medium 32^2 - 96^2, large >= 96^2 (in the original image). For a size group, objects of other sizes are ignored (neither counted as missed nor as correct), and unmatched predictions whose own area falls outside the group are ignored too.'''),
code('''cands = load_candidates("clean_640")
sz = size_study(cands, gts, conf=0.25, nms_iou=0.5); sz.to_csv(f"{RES}/size_results.csv", index=False); display(sz.round(4))
fig, ax = plt.subplots(figsize=(5.5, 3.3)); x = np.arange(3); w = .35
ax.bar(x - w/2, sz.recall[:3], w, label="recall @0.25"); ax.bar(x + w/2, sz.ap50_at_conf0_01[:3], w, label="AP@50 (all preds)")
ax.set_xticks(x); ax.set_xticklabels(sz["size"][:3]); ax.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/size_results.png", dpi=110); plt.show()'''),
md("## Experiment 4 - image conditions\n\nParameters are fixed in `src/corruptions.py`: Gaussian blur sigma 3 px; brightness x0.5 and x1.6; contrast x0.4 around the image mean; Gaussian noise sigma 25 (on 0-255); JPEG quality 10. One image under every condition:"),
code('''i = ids[7]; im0 = cv2.imread(paths[i]); names = ["clean", "blur_sigma3", "dark_x0.5", "bright_x1.6", "low_contrast_x0.4", "noise_sigma25", "jpeg_q10"]
fig, axes = plt.subplots(1, 7, figsize=(21, 3.2))
for ax, n in zip(axes, names): ax.imshow(cv2.cvtColor(CORRUPTIONS[n](im0), cv2.COLOR_BGR2RGB)); ax.set_title(n, fontsize=9); ax.axis("off")
plt.tight_layout(); plt.savefig(f"{FIG}/corruption_examples.png", dpi=100); plt.show()'''),
code('''conds = ["clean_640", "blur_sigma3_640", "dark_x0.5_640", "bright_x1.6_640", "low_contrast_x0.4_640", "noise_sigma25_640", "jpeg_q10_640"]
conds = [c_ for c_ in conds if os.path.exists(f"{CACHE}/{c_}.pkl")]; print("conditions available:", conds)
cs = condition_study(conds, gts, conf=0.25, nms_iou=0.5); cs.to_csv(f"{RES}/robustness_results.csv", index=False)
display(cs.round(4))
fig, ax = plt.subplots(figsize=(9, 3.5)); x = np.arange(len(cs)); w = .27
ax.bar(x - w, cs.recall, w, label="recall @0.25"); ax.bar(x, cs.precision, w, label="precision @0.25"); ax.bar(x + w, cs.map50, w, label="mAP@50")
ax.set_xticks(x); ax.set_xticklabels([c_.replace("_640", "") for c_ in cs.condition], rotation=25, ha="right"); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{FIG}/robustness_results.png", dpi=110); plt.show()'''),
md('''### Do the best thresholds change with the image condition?

For each condition: F1 at confidence 0.10 / 0.25 / 0.50 / 0.75 (NMS 0.5), the confidence with the best F1 on a 0.05 grid, and F1 at NMS 0.3 / 0.5 / 0.7 (confidence 0.25).'''),
code('''rows = []
for cn in conds:
    cd = load_candidates(cn); r = dict(condition=cn.replace("_640", ""))
    for t in (0.10, 0.25, 0.50, 0.75): r[f"F1@conf{t}"] = Evaluator(gts, detections(cd, t, 0.5)).metrics(0.5)["f1"]
    grid = {t: Evaluator(gts, detections(cd, t, 0.5)).metrics(0.5)["f1"] for t in np.round(np.arange(0.05, 0.96, 0.05), 2)}
    r["best_conf"] = max(grid, key=grid.get); r["best_F1"] = grid[r["best_conf"]]
    for t in (0.3, 0.5, 0.7): r[f"F1@nms{t}"] = Evaluator(gts, detections(cd, 0.25, t)).metrics(0.5)["f1"]
    rows.append(r)
tg = pd.DataFrame(rows); tg.to_csv(f"{RES}/robustness_threshold_grid.csv", index=False); display(tg.round(3))'''),
md("## Experiment 5 - input resolution\n\nClean images at 320 / 480 / 640 / 800 px (letterboxed square input). Latency was measured separately, one image at a time, one CPU thread, on this machine (`scripts/measure_latency.py`); it is a property of this hardware, not of the model in general."),
code('''res_names = [n for n in ("clean_320", "clean_480", "clean_640", "clean_800") if os.path.exists(f"{CACHE}/{n}.pkl")]
rows = []
lat = json.load(open(f"{RES}/latency.json")) if os.path.exists(f"{RES}/latency.json") else {}
for n in res_names:
    cd = load_candidates(n); m = Evaluator(gts, detections(cd, 0.25, 0.5)).metrics(0.5); a, b = full_map(cd, gts)
    s = n.split("_")[1]; L = lat.get(s, {})
    rows.append(dict(input_px=int(s), precision=m["precision"], recall=m["recall"], f1=m["f1"], mAP50=a, mAP50_95=b,
                     small_recall=Evaluator(gts, detections(cd, 0.25, 0.5)).metrics(0.5, area_range=(0, 32**2))["recall"],
                     forward_ms=L.get("forward_ms_mean"), pipeline_ms=L.get("pipeline_ms_mean")))
rs = pd.DataFrame(rows); rs.to_csv(f"{RES}/resolution_results.csv", index=False); display(rs.round(4))
fig, ax = plt.subplots(figsize=(5.5, 3.5)); ax.plot(rs.input_px, rs.mAP50, "o-", label="mAP@50"); ax.plot(rs.input_px, rs.mAP50_95, "s-", label="mAP@50:95")
ax.set_xlabel("input size (px)"); ax.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/resolution_results.png", dpi=110); plt.show()'''),
md("## Observation (computed from the tables above)"),
code('''b = cs.iloc[0]; w = cs.iloc[1:].sort_values("map50").iloc[0]; bst = cs.iloc[1:].sort_values("map50").iloc[-1]
s = sz.set_index("size"); r0, r1 = rs.iloc[0], rs.iloc[-1]; r640 = rs[rs.input_px == 640].iloc[0]
display(Markdown(f"""
**Size.** Recall at confidence 0.25 is {s.recall['small']:.3f} for small, {s.recall['medium']:.3f} for medium and {s.recall['large']:.3f} for large objects; AP@50 is {s.ap50_at_conf0_01['small']:.3f} / {s.ap50_at_conf0_01['medium']:.3f} / {s.ap50_at_conf0_01['large']:.3f}. Small objects are only {s.n_gt['small']/s.n_gt['all']:.1%} of evaluated objects here ({int(s.n_gt['small'])} objects), so small-object numbers are noisier.

**Conditions.** Change in mAP@50 against clean: {", ".join(f"{r.condition.replace('_640','')} {r.d_map50:+.3f}" for r in cs.iloc[1:].itertuples())}. Clean mAP@50 is {b.map50:.3f}. The most damaging condition tested is **{w.condition.replace('_640','')}** (mAP@50 {w.map50:.3f}, change {w.d_map50:+.3f}; recall {w.recall:.3f} vs {b.recall:.3f}); the mildest is **{bst.condition.replace('_640','')}** (mAP@50 {bst.map50:.3f}, change {bst.d_map50:+.3f}). The strength of each corruption was picked by hand (section 'Experiment 4'), so the ranking compares these specific settings, not corruption types in general.

**Resolution.** mAP@50 by input size: {", ".join(f"{int(r.input_px)} px {r.mAP50:.3f}" for r in rs.itertuples())}. The best mAP@50 is at **{int(rs.loc[rs.mAP50.idxmax(), 'input_px'])} px**, not at the largest input; mAP@50:95 peaks at {int(rs.loc[rs.mAP50_95.idxmax(), 'input_px'])} px. Recall on small objects keeps rising ({", ".join(f"{r.small_recall:.2f}" for r in rs.itertuples())}) but precision falls with size ({", ".join(f"{r.precision:.2f}" for r in rs.itertuples())}), so extra pixels buy small-object recall at the price of more false positives and about {rs.pipeline_ms.iloc[-1] / rs.pipeline_ms.iloc[0]:.1f}x the compute from the smallest to the largest input.

**Thresholds under degraded input.** Best-F1 confidence threshold per condition: {", ".join(f"{r.condition} {r.best_conf:.2f}" for r in tg.itertuples())}. For degraded inputs the best confidence moves down: the fixed default of 0.50 gives F1 {tg.set_index('condition').loc['blur_sigma3', 'F1@conf0.5']:.3f} on blur_sigma3 where the best threshold gives {tg.set_index('condition').loc['blur_sigma3', 'best_F1']:.3f}. The NMS threshold matters much less: across 0.3-0.7 the F1 of any single condition varies by at most {max(tg[['F1@nms0.3','F1@nms0.5','F1@nms0.7']].max(axis=1) - tg[['F1@nms0.3','F1@nms0.5','F1@nms0.7']].min(axis=1)):.3f}.
"""))'''),
md('''Interpretation limits: one detector, one dataset subset, one severity per corruption, one random seed for the noise. Causes ("blur removes the texture the detector relies on", "small objects have few pixels") are plausible hypotheses that these tables are consistent with or not; they would need further experiments (several severities, several seeds, other detectors) to be established.''')]
save(c, "07_robustness_size_resolution")
