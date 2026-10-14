# Object Detection Study

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
* NMS identical to `torchvision.ops.batched_nms` on 1000/1000 images at each of IoU 0.3, 0.5, 0.7.
* Decoding + filter + NMS identical to Ultralytics `predict()` on 60/60 images (max box difference 0.10 px).
* Evaluator vs `pycocotools` on the same detections: mAP@50 0.8271 vs 0.8270, mAP@50:95 0.6041 vs 0.6058 (small differences expected: COCO interpolates AP at 101 points and matches slightly differently).

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

Baseline (clean, 640 px, confidence 0.25, NMS 0.5): precision **0.710**, recall **0.837**, F1 0.769; mAP@50 **0.827**, mAP@50:95 **0.604**. Per-class AP: `results/per_class_ap_clean640.csv`. All experiment tables are combined in `results/detection_results.csv`; write-ups are in `experiments/`.

Object size (confidence 0.25, NMS 0.5):

| size   |   n_gt |   precision |   recall |   ap50_at_conf0_01 |
|:-------|-------:|------------:|---------:|-------------------:|
| small  |    103 |       0.241 |    0.757 |              0.615 |
| medium |    758 |       0.676 |    0.802 |              0.744 |
| large  |   1561 |       0.823 |    0.859 |              0.864 |
| all    |   2422 |       0.71  |    0.837 |              0.827 |

Only 103 small objects exist in this subset, so the small-object rows are noisy.

Input resolution (latency: 1 CPU thread, this machine):

|   input_px |   precision |   recall |    f1 |   mAP50 |   mAP50_95 |   small_recall |   forward_ms |   pipeline_ms |
|-----------:|------------:|---------:|------:|--------:|-----------:|---------------:|-------------:|--------------:|
|        320 |       0.762 |    0.75  | 0.756 |   0.792 |      0.575 |          0.33  |       55.15  |        63.902 |
|        480 |       0.739 |    0.828 | 0.781 |   0.84  |      0.624 |          0.602 |      100.114 |       105.525 |
|        640 |       0.71  |    0.837 | 0.769 |   0.827 |      0.604 |          0.757 |      178.125 |       186.464 |
|        800 |       0.679 |    0.827 | 0.746 |   0.802 |      0.551 |          0.777 |      266.974 |       279.077 |

mAP is best at 480 px on this subset; going to 640 or 800 px costs more time and does not improve it. Larger inputs raise small-object recall and add false positives.

## 10. Threshold experiments

Confidence threshold (NMS 0.5):

|   conf |   dets_per_img |   precision |   recall |    f1 |   map50_truncated |
|-------:|---------------:|------------:|---------:|------:|------------------:|
|   0.1  |          4.695 |       0.503 |    0.898 | 0.645 |             0.803 |
|   0.25 |          3.132 |       0.71  |    0.837 | 0.769 |             0.764 |
|   0.5  |          2.177 |       0.879 |    0.732 | 0.798 |             0.679 |
|   0.75 |          1.278 |       0.965 |    0.491 | 0.651 |             0.473 |

(`map50_truncated` = AP computed from only the boxes above the threshold.) F1 is highest at 0.50; the usual default of 0.25 is a more recall-heavy choice.

NMS IoU threshold (confidence 0.25):

| nms_iou   |   dets_per_img |   precision |   recall |    f1 |   duplicate_fp |   crowded_person_recall |   map50_conf0_01 |
|:----------|---------------:|------------:|---------:|------:|---------------:|------------------------:|-----------------:|
| 0.3       |          3.03  |       0.723 |    0.823 | 0.769 |              1 |                   0.856 |            0.811 |
| 0.5       |          3.132 |       0.71  |    0.837 | 0.769 |             13 |                   0.894 |            0.827 |
| 0.7       |          3.33  |       0.672 |    0.841 | 0.747 |            131 |                   0.899 |            0.822 |
| off       |         25.019 |       0.089 |    0.847 | 0.162 |          16636 |                   0.913 |            0.211 |

NMS is essential (precision 0.089 without it). 0.7 leaves 131 duplicate false positives; 0.3 removes almost all duplicates but lowers person recall in crowded scenes (0.856 vs 0.894 at 0.5; 63 images, so treat as indicative). Full discussion: `experiments/confidence_threshold.md`, `experiments/nms_threshold.md`.

## 11. Robustness to image conditions

Same detector, same images, confidence 0.25, NMS 0.5, one severity per corruption:

| condition         |   precision |   recall |    f1 |   map50 |   map50_95 |   d_map50 |
|:------------------|------------:|---------:|------:|--------:|-----------:|----------:|
| clean             |       0.71  |    0.837 | 0.769 |   0.827 |      0.604 |     0     |
| blur_sigma3       |       0.727 |    0.543 | 0.622 |   0.561 |      0.365 |    -0.266 |
| dark_x0.5         |       0.724 |    0.833 | 0.775 |   0.823 |      0.6   |    -0.004 |
| bright_x1.6       |       0.704 |    0.809 | 0.753 |   0.803 |      0.576 |    -0.024 |
| low_contrast_x0.4 |       0.733 |    0.784 | 0.758 |   0.791 |      0.565 |    -0.036 |
| noise_sigma25     |       0.749 |    0.599 | 0.666 |   0.611 |      0.406 |    -0.216 |
| jpeg_q10          |       0.603 |    0.29  | 0.391 |   0.287 |      0.167 |    -0.54  |

Blur, noise and especially JPEG quality 10 hurt a lot; darkening hardly matters. Precision stays about the same while recall drops, so the detector returns fewer boxes instead of more wrong ones. The JPEG drop was verified independently with Ultralytics' own `predict()` (`results/jpeg_sanity_check.json`).

Best confidence threshold per condition (by F1): clean 0.50, blur_sigma3 0.20, dark_x0.5 0.40, bright_x1.6 0.45, low_contrast_x0.4 0.35, noise_sigma25 0.25, jpeg_q10 0.20. A threshold tuned on clean images discards too many correct detections on degraded ones; the best NMS threshold barely changes (F1 moves by at most 0.023 between 0.3 and 0.7). Details: `experiments/robustness.md`.

## 12. Error analysis

At the baseline setting, 826 false positives and 395 missed objects were each assigned a type (`src/analysis.py`, `results/failure_cases/error_log.csv`; gallery `figures/failure_cases.png`).

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

## 14. What I learned

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
