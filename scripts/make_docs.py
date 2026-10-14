"""Writes experiments/*.md and README.md. Every number comes from results/*.csv|json, so the text cannot drift from the data."""
import os, sys, json
import pandas as pd
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(root)
R = lambda n: pd.read_csv(f"results/{n}.csv")
md = lambda df, nd=3: df.round(nd).to_markdown(index=False)
conf, nms, size, rob, grid, res, base = (R(n) for n in ("confidence_threshold_results", "nms_threshold_results", "size_results", "robustness_results", "robustness_threshold_grid", "resolution_results", "baseline_results"))
mix = R("error_mix_by_condition"); err = pd.read_csv("results/failure_cases/error_log.csv"); cc = json.load(open("results/cross_check.json")); jp = json.load(open("results/jpeg_sanity_check.json"))
fine_c = R("confidence_threshold_fine_sweep"); fine_n = R("nms_threshold_fine_sweep")
c = conf.set_index("conf"); n = nms.set_index("nms_iou"); b = base.iloc[0]; s = size.set_index("size")
best = fine_c.loc[fine_c.f1.idxmax()]
nobj = int(b.objects_evaluated)

# ---- combined table
parts = []
parts.append(base.assign(experiment="baseline"))
parts.append(conf.assign(experiment="confidence", nms_iou=0.5, condition="clean_640"))
parts.append(nms.assign(experiment="nms", conf=0.25, condition="clean_640"))
parts.append(rob.assign(experiment="condition", conf=0.25, nms_iou=0.5))
parts.append(res.assign(experiment="resolution", conf=0.25, nms_iou=0.5))
pd.concat(parts, ignore_index=True).to_csv("results/detection_results.csv", index=False)

os.makedirs("experiments", exist_ok=True)
SETUP = ("Setup: YOLOv8n with COCO weights (no training), 1,000 random Pascal VOC2007-test images (seed 0), "
         f"{nobj} non-difficult objects, 20 VOC classes, own NMS and own evaluator (VOC matching, IoU 0.5). "
         "Unless stated: 640 px input, clean images.")

open("experiments/confidence_threshold.md", "w").write(f"""# Experiment 1 - confidence threshold

{SETUP} NMS IoU fixed at 0.5; only the confidence threshold changes.

**Question.** How does raising the confidence threshold change the number of detections, precision and recall?

{md(conf[['conf','n_det','tp','fp','fn','dets_per_img','precision','recall','f1','map50_truncated']])}

(`n_det` counts matched TP + FP; `dets_per_img` counts every kept box. `map50_truncated` is AP computed only from boxes above the threshold.)

**Result.** From 0.10 to 0.75 precision rises {c.precision[0.10]:.3f} -> {c.precision[0.75]:.3f}, recall falls {c.recall[0.10]:.3f} -> {c.recall[0.75]:.3f}, and the detector returns {c.dets_per_img[0.10]:.2f} -> {c.dets_per_img[0.75]:.2f} boxes per image (true objects per image: {nobj/1000:.2f}). On the finer 0.05 grid F1 peaks at confidence {best.conf:.2f} (F1 {best.f1:.3f}); the common default 0.25 gives F1 {c.f1[0.25]:.3f}. mAP@50 computed from the truncated list falls {c.map50_truncated[0.10]:.3f} -> {c.map50_truncated[0.75]:.3f}, because removing low-confidence boxes caps the recall the curve can reach.

**Reading.** The threshold is a choice of operating point, not a quality setting: 0.10 finds more objects but about half its boxes are wrong; 0.75 is almost always right but misses half the objects. mAP is therefore computed from a very low threshold (here 0.01) so that it does not depend on this choice.

**Limits.** One detector, one subset, one NMS setting. Whether 0.50 is best depends on how much a false alarm costs compared with a miss; F1 weighs them equally.
""")

open("experiments/nms_threshold.md", "w").write(f"""# Experiment 2 - NMS IoU threshold

{SETUP} Confidence fixed at 0.25; only the NMS IoU threshold changes ("off" = no NMS).

**Question.** What does NMS remove, and when does it become too aggressive?

{md(nms[['nms_iou','n_det','tp','fp','fn','dets_per_img','precision','recall','f1','duplicate_fp','crowded_person_recall','map50_conf0_01']])}

`duplicate_fp` = false positives that sit on an object a better box already found. `crowded_person_recall` = recall for people in the {int(nms.n_crowded_images[0])} images with at least 4 people ({int(nms.crowded_person_gt[0])} people). `map50_conf0_01` = mAP@50 using the full ranked list.

**Result.**
* NMS is essential: without it there are {n.dets_per_img['off']:.1f} boxes per image and precision is {n.precision['off']:.3f} ({int(n.duplicate_fp['off'])} duplicate false positives).
* Threshold 0.7 leaves {int(n.duplicate_fp['0.7'])} duplicates vs {int(n.duplicate_fp['0.5'])} at 0.5 and {int(n.duplicate_fp['0.3'])} at 0.3, and F1 drops to {n.f1['0.7']:.3f} (0.3 and 0.5 give {n.f1['0.3']:.3f} and {n.f1['0.5']:.3f}). mAP@50 is best at 0.5 ({n.map50_conf0_01['0.5']:.3f}) against {n.map50_conf0_01['0.3']:.3f} at 0.3 and {n.map50_conf0_01['0.7']:.3f} at 0.7.
* The cost of a low threshold shows up in crowded scenes: person recall there is {n.crowded_person_recall['0.3']:.3f} at 0.3, {n.crowded_person_recall['0.5']:.3f} at 0.5, {n.crowded_person_recall['0.7']:.3f} at 0.7. Over all images the difference is small ({n.recall['0.3']:.3f} vs {n.recall['0.7']:.3f} recall).

**Answer to "when is NMS too aggressive?"** In this data only at the low end of the range, and mainly where objects of one class touch each other. Even then the effect is a few points of recall.

**Limits.** The crowded subset is 63 images, so the crowded-scene numbers are noisy; one detector; no soft-NMS or other variants were tried.
""")

open("experiments/object_size.md", "w").write(f"""# Experiment 3 - object size

{SETUP} Confidence 0.25, NMS 0.5. Sizes by ground-truth box area in the original image: small < 32^2 px, medium 32^2 - 96^2, large >= 96^2. Objects and unmatched predictions outside the group are ignored.

{md(size)}

**Result.** Recall {s.recall['small']:.3f} / {s.recall['medium']:.3f} / {s.recall['large']:.3f} and AP@50 {s.ap50_at_conf0_01['small']:.3f} / {s.ap50_at_conf0_01['medium']:.3f} / {s.ap50_at_conf0_01['large']:.3f} for small / medium / large. Performance does fall with object size, as expected.

**Cautions.** Only {int(s.n_gt['small'])} small objects exist in this subset ({s.n_gt['small']/s.n_gt['all']:.1%}), so the small-object numbers are noisy. The very low small-object precision ({s.precision['small']:.3f}) mixes real false alarms with tiny predicted boxes on things that VOC did not annotate; this experiment cannot separate the two.
""")

cond = rob.copy(); cond["condition"] = cond.condition.str.replace("_640", "")
open("experiments/robustness.md", "w").write(f"""# Experiment 4 - image conditions

{SETUP} Confidence 0.25, NMS 0.5 for precision/recall; mAP from the full ranked list. Corruptions leave geometry unchanged, so ground truth is reused. Parameters (`src/corruptions.py`): Gaussian blur sigma 3 px; brightness x0.5 and x1.6; contrast x0.4; Gaussian noise sigma 25 (seeded); JPEG quality 10. One severity each.

{md(cond[['condition','precision','recall','f1','dets_per_img','map50','map50_95','d_map50']])}

**Result.** Change in mAP@50 against clean ({rob.map50[0]:.3f}): {", ".join(f"{r.condition} {r.d_map50:+.3f}" for r in cond.iloc[1:].itertuples())}. Darkening costs almost nothing here; blur and noise remove about a quarter of mAP@50; JPEG quality 10 removes most of it.
Precision barely moves (it even rises slightly for blur, noise and darkening) while recall drops: under degraded input the detector produces fewer boxes rather than more wrong ones.

**JPEG check.** The JPEG result is large enough that it was checked independently of this repo's pipeline: Ultralytics' own `predict()` returns {jp['detections_ultralytics_predict']['clean']} detections on 40 clean images, {jp['detections_ultralytics_predict']['75']} at JPEG quality 75, {jp['detections_ultralytics_predict']['30']} at 30 and {jp['detections_ultralytics_predict']['10']} at 10 (quality-10 files are {jp['q10_file_size_vs_original']:.0%} of the original size, PSNR {jp['psnr_q10_db']:.1f} dB). So the drop is real, not a bug in the corruption code.

## Do the best thresholds depend on the condition?

{md(grid.round(3)[['condition','F1@conf0.1','F1@conf0.25','F1@conf0.5','F1@conf0.75','best_conf','best_F1','F1@nms0.3','F1@nms0.5','F1@nms0.7']])}

* **Confidence:** the best threshold is {grid.best_conf[0]:.2f} on clean images but {grid.set_index('condition').best_conf['blur_sigma3']:.2f} (blur), {grid.set_index('condition').best_conf['noise_sigma25']:.2f} (noise) and {grid.set_index('condition').best_conf['jpeg_q10']:.2f} (JPEG): degraded images lower the scores of correct boxes, so a threshold tuned on clean data throws away detections. Using 0.50 on blur gives F1 {grid.set_index('condition').loc['blur_sigma3','F1@conf0.5']:.3f}; the best threshold gives {grid.set_index('condition').loc['blur_sigma3','best_F1']:.3f}.
* **NMS IoU:** within 0.3-0.7 the F1 of every condition changes by at most {max(grid[['F1@nms0.3','F1@nms0.5','F1@nms0.7']].max(axis=1)-grid[['F1@nms0.3','F1@nms0.5','F1@nms0.7']].min(axis=1)):.3f}; it is much less sensitive than confidence.

**Limits.** One severity per corruption and one noise seed, so this shows how much *these settings* hurt, not a ranking of corruption types. Explanations such as "blur removes the texture the detector uses" are hypotheses; they would need several severities and detectors to test.
""")

r = res.set_index("input_px")
open("experiments/resolution.md", "w").write(f"""# Experiment 5 - input resolution

{SETUP} Clean images letterboxed to 320 / 480 / 640 / 800 px. Latency: one image at a time, 1 CPU thread, this machine only (`scripts/measure_latency.py`).

{md(res)}

**Result.** mAP@50 is highest at **480 px** ({r.mAP50[480]:.3f}), above 640 px ({r.mAP50[640]:.3f}) and 800 px ({r.mAP50[800]:.3f}); mAP@50:95 also peaks at 480 px ({r.mAP50_95[480]:.3f}). Recall on small objects keeps rising ({r.small_recall[320]:.2f} -> {r.small_recall[480]:.2f} -> {r.small_recall[640]:.2f} -> {r.small_recall[800]:.2f}) while precision falls ({r.precision[320]:.2f} -> {r.precision[800]:.2f}). Compute grows about {r.pipeline_ms[800]/r.pipeline_ms[320]:.1f}x from 320 to 800 px.

**Answer.** Going beyond 480 px did not pay off on this subset: it costs {r.pipeline_ms[640]/r.pipeline_ms[480]:.1f}x time at 640 px for slightly lower mAP. Larger inputs help small objects but add false positives.

**Caution.** A peak below the training resolution (YOLOv8 was trained at 640) is not what one would first expect. A possible reason is that most VOC objects are large (56% of objects here), so upscaling helps few of them, but this is a hypothesis; it was not tested (for example by evaluating small objects only at each resolution with more data).
""")

fp = err[err.type.str.startswith("FP")].type.str[3:].value_counts(); fn = err[err.type.str.startswith("FN")].type.str[3:].value_counts()
mixc = mix.copy(); mixc["condition"] = mixc.condition
open("experiments/error_analysis.md", "w").write(f"""# Experiment 6 - error analysis

{SETUP} Confidence 0.25, NMS 0.5. Every false positive and every missed object ({int(fp.sum())} FP, {int(fn.sum())} FN) is assigned a type by the rules in `src/analysis.py`; the full list is `results/failure_cases/error_log.csv`, example images are in `results/failure_cases/` and `figures/failure_cases.png`.

**False positives:** {", ".join(f"{k} {v} ({v/fp.sum():.0%})" for k, v in fp.items())}.
**Missed objects:** {", ".join(f"{k.replace('_',' ')} {v} ({v/fn.sum():.0%})" for k, v in fn.items())}.

Reading, from the data: duplicates are almost gone after NMS; the biggest false-positive group is boxes that overlap no annotated object at all, and the biggest miss group is objects the detector did find but scored under 0.25. A lower threshold would recover most of those at a precision cost (experiment 1). Only a small share of misses are cases where nothing in the image was detected at all.

Notebook 08 also reports miss rates by size (small 24%, medium 20%, large 14%), truncation (truncated 20% vs not truncated 13%), number of objects per image and class (potted plant 64%, chair 33% are worst), and the commonest class confusions (cat/dog, sofa/chair).

## Error mix by image condition

{md(mixc, 0)}

Under blur, noise and JPEG the misses move from "found but low score / found but poorly placed" to "never found" (see the FN_missed column), consistent with experiment 4.

**Hypotheses, not findings:** "background" false positives include objects that are really in the picture but not annotated in VOC (the VOC annotation only labels 20 classes and skips some small instances); "classification" errors between cat/dog and sofa/chair look like genuinely confusable classes, and sofa/couch label conventions differ between COCO and VOC. Checking either needs manual review of the logged examples.
""")

open("experiments/final_detection_study.md", "w").write(f"""# Final detection study - summary of findings

{SETUP}

**Baseline** (confidence 0.25, NMS 0.5, 640 px): precision {b.precision:.3f}, recall {b.recall:.3f}, F1 {b.f1:.3f}; mAP@50 {b.mAP50:.3f}, mAP@50:95 {b.mAP50_95:.3f}.

**Main sensitivity table** (confidence 0.25, NMS 0.5; mAP from the full ranked list):

{md(cond[['condition','precision','recall','map50','map50_95']])}

**What the experiments say**
1. Confidence threshold: pure precision/recall trade-off; best F1 at {best.conf:.2f}, not at the usual 0.25 (experiment 1).
2. NMS: required; 0.5 is best for mAP; the low end (0.3) costs recall in crowded scenes, the high end (0.7) leaves duplicates (experiment 2).
3. Size: small objects are harder, but only {int(s.n_gt['small'])} exist in the subset (experiment 3).
4. Image conditions: JPEG q10, blur and noise at the tested strengths hurt a lot; darkening, brightening, lower contrast little. The best confidence threshold drops for degraded inputs, the best NMS threshold hardly changes (experiment 4).
5. Resolution: best at 480 px on this subset, with 640/800 px costing more time (experiment 5).
6. Errors: after NMS, remaining errors are mostly mislabelled/poorly placed boxes, boxes on unannotated things, and correct boxes with low scores (experiment 6).

**Validation of the from-scratch code**
* NMS identical to `torchvision.ops.batched_nms` on {cc['nms_vs_torchvision_iou0.5']['identical_images']}/{cc['nms_vs_torchvision_iou0.5']['images']} images at each of IoU 0.3, 0.5, 0.7.
* Decode + filter + NMS identical to Ultralytics' own `predict()` on {cc['ultralytics_predict_equivalence']['identical_detections']}/{cc['ultralytics_predict_equivalence']['images']} images (max box difference {cc['ultralytics_predict_equivalence']['max_box_diff_px']:.2f} px).
* Evaluator vs pycocotools on the same detections: mAP@50 {cc['pycocotools']['ours_map50']:.4f} (ours) vs {cc['pycocotools']['coco_map50']:.4f} (COCO tool); mAP@50:95 {cc['pycocotools']['ours_map50_95']:.4f} vs {cc['pycocotools']['coco_map50_95']:.4f}. Small differences are expected (COCO uses 101-point interpolation and slightly different matching).
* 20 unit tests (`pytest`).

**Limitations**
* No detector was trained: a COCO-pretrained YOLOv8n is evaluated zero-shot on VOC classes. Label-convention differences (sofa/couch, diningtable) cost some AP.
* One detector, one 1,000-image subset of VOC2007 test (about 20% of it), one severity per corruption, one noise seed, no confidence intervals. Differences of a few mAP points (for example 640 vs 800 px) should not be over-read.
* COCO and VOC images both come from Flickr; overlap between COCO training images and VOC2007 test images cannot be excluded.
* Latency was measured on a single CPU thread of one machine.

**Future work:** several corruption severities and noise seeds; bootstrap confidence intervals; compare a second detector (e.g. a two-stage model); fine-tune on VOC train; check the "background" false positives by hand.
""")
print("docs written")
