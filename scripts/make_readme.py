import os, json
import pandas as pd
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(root)
R = lambda n: pd.read_csv(f"results/{n}.csv")
md = lambda df, nd=3: df.round(nd).to_markdown(index=False)
base, conf, nms, rob, res, size, grid = (R(n) for n in ("baseline_results", "confidence_threshold_results", "nms_threshold_results", "robustness_results", "resolution_results", "size_results", "robustness_threshold_grid"))
cc = json.load(open("results/cross_check.json")); b = base.iloc[0]
rob["condition"] = rob.condition.str.replace("_640", "")
txt = f"""# Object Detection Study

Bounding boxes, IoU, NMS and detection metrics written from scratch with NumPy, then used to study a pretrained YOLOv8n detector on Pascal VOC: how confidence and NMS thresholds, object size, input resolution and image corruptions change what the detector finds and where it fails.

Part of my computer vision portfolio (Month 4). **No model is trained here.** The detector is a COCO-pretrained YOLOv8n used as it is; the work is in the evaluation code and the experiments around it.

## 1. Problem

Classification says what is in an image. Detection also has to say where, and has to be judged on both: a box can be in the right class and in the wrong place, or fit perfectly and carry the wrong label. This project builds the pieces that decide that (box geometry, IoU, NMS, TP/FP/FN matching, precision, recall, AP, mAP) and uses them to look closely at a modern detector instead of only running it.

## 2. Research question

How do IoU, confidence thresholds and NMS affect detection quality, and how do input size, resolution and image quality change the picture? Sub-questions: what does NMS remove and when does it remove too much; how much does each threshold move precision and recall; do small objects fail more; which image degradations hurt; where do the remaining errors come from.

## 3. Background

* One-stage detectors (YOLO) predict boxes and class scores at thousands of positions in one pass; neighbouring positions fire on the same object, so a post-processing step (NMS) is needed.
* A prediction is correct if the class matches and IoU with an unmatched ground-truth box is at least a threshold (0.5 for mAP@50).
* mAP@50:95 averages over IoU thresholds 0.50-0.95 and rewards tight boxes.

## 4. Detection fundamentals (one notebook each)

| notebook | topic |
|---|---|
| [01](notebooks/01_bounding_boxes.ipynb) | box formats, conversions, typical coordinate mistakes, dataset summary |
| [02](notebooks/02_iou_from_scratch.ipynb) | IoU by hand, tests, IoU vs shift/scale, IoU of real detections |
| [03](notebooks/03_nms_from_scratch.ipynb) | NMS trace, thresholds 0.3/0.5/0.7, real and crowded images, class-aware vs agnostic |
| [04](notebooks/04_detection_metrics.ipynb) | TP/FP/FN, precision/recall, PR curve, AP by hand, mAP@50 vs @50:95, per-class AP |
| [05](notebooks/05_yolo_baseline.ipynb) | what the detector outputs, equivalence with Ultralytics, baseline numbers |
| [06](notebooks/06_threshold_experiments.ipynb) | confidence and NMS threshold experiments |
| [07](notebooks/07_robustness_size_resolution.ipynb) | size, image conditions, resolution |
| [08](notebooks/08_error_analysis.ipynb) | false positive / missed-object taxonomy and gallery |

## 5. What I implemented myself vs what comes from a framework

| written here (`src/`) | from a library |
|---|---|
| `boxes.py` conversions, area, intersection, union | network weights and forward pass (Ultralytics `yolov8n.pt`, PyTorch) |
| `iou.py` scalar and vectorised IoU | image reading/resizing (OpenCV) |
| `nms.py` confidence filter, NMS (loop and vectorised), per-class NMS | |
| `metrics.py` precision, recall, PR curve, AP (all-point and 11-point) | |
| `evaluation.py` VOC-style matching, per-class AP, mAP@50:95, size breakdown | |
| `model.py` letterbox, raw-output decoding, VOC-COCO class mapping | |
| `analysis.py` error taxonomy; `corruptions.py` | |

Checks that the from-scratch code is right (`scripts/cross_check.py`, `tests/`):

* 20 unit tests with hand-computed values (IoU edge cases, NMS threshold direction, AP on a small curve, VOC matching rules).
* NMS identical to `torchvision.ops.batched_nms` on {cc['nms_vs_torchvision_iou0.5']['identical_images']}/{cc['nms_vs_torchvision_iou0.5']['images']} images at each of IoU 0.3, 0.5, 0.7.
* Decoding + filter + NMS identical to Ultralytics `predict()` on {cc['ultralytics_predict_equivalence']['identical_detections']}/{cc['ultralytics_predict_equivalence']['images']} images (max box difference {cc['ultralytics_predict_equivalence']['max_box_diff_px']:.2f} px).
* Evaluator vs `pycocotools` on the same detections: mAP@50 {cc['pycocotools']['ours_map50']:.4f} vs {cc['pycocotools']['coco_map50']:.4f}, mAP@50:95 {cc['pycocotools']['ours_map50_95']:.4f} vs {cc['pycocotools']['coco_map50_95']:.4f} (small differences expected: COCO interpolates AP at 101 points and matches slightly differently).

## 6. Dataset

Pascal VOC 2007 **test**, random subset of 1,000 images (seed 0): 3,004 annotated objects, 582 flagged *difficult* (ignored as in the VOC protocol), 2,422 evaluated, 20 classes. Details and download commands: [data/README.md](data/README.md). There is no training or validation split because nothing is trained or tuned.

## 7. Modern detector

YOLOv8n, official COCO weights, 3.2 M parameters, used without any fine-tuning. It predicts the 80 COCO classes; each VOC class is mapped to its COCO counterpart (e.g. `sofa` -> `couch`, `tvmonitor` -> `tv`) and the other 60 classes are dropped. Letterboxing, decoding, thresholds and NMS are done by this repo's code (section 5).

## 8. Experimental setup

* Input 640 px unless stated; default operating point confidence 0.25, NMS IoU 0.5; matching at IoU 0.5.
* Raw candidates (confidence >= 0.01) are cached once per condition (`scripts/run_inference.py`); every experiment re-runs filtering, NMS and evaluation offline on the cache.
* mAP uses all predictions with confidence >= 0.01; precision/recall are at the stated confidence threshold.
* Hardware: a single CPU thread on a shared cloud machine. Latency numbers are specific to it.
* Everything is reproducible with the commands in section 15.

## 9. Results

Baseline (clean, 640 px, confidence 0.25, NMS 0.5): precision **{b.precision:.3f}**, recall **{b.recall:.3f}**, F1 {b.f1:.3f}; mAP@50 **{b.mAP50:.3f}**, mAP@50:95 **{b.mAP50_95:.3f}**. Per-class AP: `results/per_class_ap_clean640.csv`. All experiment tables are combined in `results/detection_results.csv`; write-ups are in `experiments/`.

Object size (confidence 0.25, NMS 0.5):

{md(size)}

Only {int(size.n_gt[0])} small objects exist in this subset, so the small-object rows are noisy.

Input resolution (latency: 1 CPU thread, this machine):

{md(res)}

mAP is best at 480 px on this subset; going to 640 or 800 px costs more time and does not improve it. Larger inputs raise small-object recall and add false positives.

## 10. Threshold experiments

Confidence threshold (NMS 0.5):

{md(conf[['conf','dets_per_img','precision','recall','f1','map50_truncated']])}

(`map50_truncated` = AP computed from only the boxes above the threshold.) F1 is highest at 0.50; the usual default of 0.25 is a more recall-heavy choice.

NMS IoU threshold (confidence 0.25):

{md(nms[['nms_iou','dets_per_img','precision','recall','f1','duplicate_fp','crowded_person_recall','map50_conf0_01']])}

NMS is essential (precision {nms.precision.iloc[3]:.3f} without it). 0.7 leaves {int(nms.duplicate_fp.iloc[2])} duplicate false positives; 0.3 removes almost all duplicates but lowers person recall in crowded scenes ({nms.crowded_person_recall.iloc[0]:.3f} vs {nms.crowded_person_recall.iloc[1]:.3f} at 0.5; 63 images, so treat as indicative). Full discussion: `experiments/confidence_threshold.md`, `experiments/nms_threshold.md`.

## 11. Robustness to image conditions

Same detector, same images, confidence 0.25, NMS 0.5, one severity per corruption:

{md(rob[['condition','precision','recall','f1','map50','map50_95','d_map50']])}

Blur, noise and especially JPEG quality 10 hurt a lot; darkening hardly matters. Precision stays about the same while recall drops, so the detector returns fewer boxes instead of more wrong ones. The JPEG drop was verified independently with Ultralytics' own `predict()` (`results/jpeg_sanity_check.json`).

Best confidence threshold per condition (by F1): {", ".join(f"{r.condition} {r.best_conf:.2f}" for r in grid.itertuples())}. A threshold tuned on clean images discards too many correct detections on degraded ones; the best NMS threshold barely changes (F1 moves by at most {max(grid[['F1@nms0.3','F1@nms0.5','F1@nms0.7']].max(axis=1)-grid[['F1@nms0.3','F1@nms0.5','F1@nms0.7']].min(axis=1)):.3f} between 0.3 and 0.7). Details: `experiments/robustness.md`.

## 12. Error analysis

At the baseline setting, {int((pd.read_csv('results/failure_cases/error_log.csv').type.str.startswith('FP')).sum())} false positives and {int((pd.read_csv('results/failure_cases/error_log.csv').type.str.startswith('FN')).sum())} missed objects were each assigned a type (`src/analysis.py`, `results/failure_cases/error_log.csv`; gallery `figures/failure_cases.png`).

* False positives: mostly boxes that overlap no annotated object (59%), then poorly placed boxes (27%), wrong-class boxes (13%), and duplicates (2%).
* Missed objects: most were actually found but scored under 0.25 (39%) or found with a poorly fitting box (31%); 13% were not found at all, 13% were found with the wrong class, 3% were removed by NMS.
* Miss rate rises as objects get smaller (24% small, 20% medium, 14% large) and for truncated objects (20% vs 13%); potted plants (64%) and chairs (33%) are the weakest classes; frequent confusions are cat/dog and sofa/chair.
* Under blur, noise and JPEG the share of "never found" misses grows sharply.

These are descriptions of what the boxes show. Reasons (unannotated objects counted as false positives, VOC/COCO label conventions) are hypotheses.

## 13. Limitations

* No training: a COCO-pretrained model is evaluated zero-shot on VOC classes, so label-convention differences cost some AP.
* One detector, one 1,000-image subset (about 20% of VOC2007 test), one severity per corruption, one noise seed, no confidence intervals. Differences of a few mAP points (e.g. 640 vs 800 px) are within what a different subset could change.
* COCO and VOC images both come from Flickr; overlap between COCO training images and VOC test images cannot be ruled out.
* Latency is for one CPU thread on one machine.

## 14. What I learned (draft - rewrite in your own words)

* IoU is stricter than it looks: a box of half the area, fully inside the ground truth, scores only 0.5; and mAP@50:95 is about 22 points below mAP@50 for this detector.
* The confidence threshold is an operating point, not a quality knob; mAP is computed with a very low threshold so that it does not depend on that choice.
* NMS decides more than I expected: without it precision is below 0.1. Its threshold matters in crowded scenes but much less than the confidence threshold.
* Image conditions mostly lower scores and recall rather than adding false alarms, which is why a confidence threshold tuned on clean images is wrong for degraded ones.
* More input resolution is not automatically better.
* Error categories (found-but-low-score vs never-found) point to different fixes, which a single mAP number hides.

## 15. Reproduce

```bash
pip install -r requirements.txt
# data and weights: see data/README.md
python scripts/run_inference.py --voc data/VOCdevkit/VOC2007 --n 1000 --seed 0   # ~35 min on 1 CPU; writes results/cache
python scripts/measure_latency.py data/VOCdevkit/VOC2007
python scripts/cross_check.py data/VOCdevkit/VOC2007
python scripts/jpeg_sanity_check.py data/VOCdevkit/VOC2007
python scripts/build_notebooks.py --run        # builds and executes all notebooks
python scripts/make_docs.py && python scripts/make_readme.py
pytest
```

The cached detector output (`results/cache/`) is included, so notebooks 02-04 and 06-08 run without the dataset images except where they draw pictures.

## 16. Future work

Several corruption severities and noise seeds with bootstrap confidence intervals; a second detector (e.g. a two-stage model); fine-tuning on VOC train; soft-NMS and class-agnostic NMS variants; manual review of the "background" false positives to see how many are unannotated real objects.

## 17. References

* Redmon et al., *You Only Look Once: Unified, Real-Time Object Detection*, CVPR 2016.
* Jocher et al., Ultralytics YOLOv8, https://github.com/ultralytics/ultralytics.
* Everingham et al., *The Pascal Visual Object Classes (VOC) Challenge*, IJCV 2010.
* Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014.
* Neubeck and Van Gool, *Efficient Non-Maximum Suppression*, ICPR 2006.
* Ren et al., *Faster R-CNN*, NeurIPS 2015.

## Repository layout

```
src/            boxes, iou, nms, metrics, evaluation, dataset, model, corruptions, analysis, experiments, visualization
notebooks/      01-08, executed with outputs
experiments/    written findings per experiment + final summary
results/        CSV tables, cache/ (raw detector output), failure_cases/, cross_check.json, latency.json
figures/        saved figures
scripts/        run_inference, measure_latency, cross_check, jpeg_sanity_check, build_notebooks, make_docs, make_readme
tests/          unit tests
```
"""
open("README.md", "w").write(txt); print("README written", len(txt))
