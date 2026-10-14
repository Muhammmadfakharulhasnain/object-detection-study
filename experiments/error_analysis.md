# Experiment 6 - error analysis

Setup: YOLOv8n with COCO weights (no training), 1,000 random Pascal VOC2007-test images (seed 0), 2422 non-difficult objects, 20 VOC classes, own NMS and own evaluator (VOC matching, IoU 0.5). Unless stated: 640 px input, clean images. Confidence 0.25, NMS 0.5. Every false positive and every missed object (826 FP, 395 FN) is assigned a type by the rules in `src/analysis.py`; the full list is `results/failure_cases/error_log.csv`, example images are in `results/failure_cases/` and `figures/failure_cases.png`.

**False positives:** background 485 (59%), localization 220 (27%), classification 108 (13%), duplicate 13 (2%).
**Missed objects:** low confidence 154 (39%), localization 124 (31%), missed 53 (13%), classification 52 (13%), nms suppressed 12 (3%).

Reading, from the data: duplicates are almost gone after NMS; the biggest false-positive group is boxes that overlap no annotated object at all, and the biggest miss group is objects the detector did find but scored under 0.25. A lower threshold would recover most of those at a precision cost (experiment 1). Only a small share of misses are cases where nothing in the image was detected at all.

Notebook 08 also reports miss rates by size (small 24%, medium 20%, large 14%), truncation (truncated 20% vs not truncated 13%), number of objects per image and class (potted plant 64%, chair 33% are worst), and the commonest class confusions (cat/dog, sofa/chair).

## Error mix by image condition

| condition         |   FP_duplicate |   FP_classification |   FP_localization |   FP_background |   FN_classification |   FN_localization |   FN_nms_suppressed |   FN_low_confidence |   FN_missed |
|:------------------|---------------:|--------------------:|------------------:|----------------:|--------------------:|------------------:|--------------------:|--------------------:|------------:|
| clean             |             13 |                 108 |               220 |             485 |                  52 |               124 |                  12 |                 154 |          53 |
| blur_sigma3       |              8 |                 154 |               166 |             166 |                 105 |               163 |                   6 |                 494 |         339 |
| dark_x0.5         |              7 |                 100 |               218 |             443 |                  53 |               127 |                  10 |                 157 |          58 |
| bright_x1.6       |             10 |                 120 |               217 |             475 |                  45 |               125 |                  12 |                 209 |          72 |
| low_contrast_x0.4 |              6 |                 120 |               209 |             356 |                  69 |               130 |                  10 |                 224 |          90 |
| noise_sigma25     |              7 |                 136 |               151 |             191 |                  85 |               157 |                   8 |                 412 |         309 |
| jpeg_q10          |              5 |                 135 |               129 |             194 |                 123 |               151 |                   9 |                 533 |         904 |

Under blur, noise and JPEG the misses move from "found but low score / found but poorly placed" to "never found" (see the FN_missed column), consistent with experiment 4.

**Hypotheses, not findings:** "background" false positives include objects that are really in the picture but not annotated in VOC (the VOC annotation only labels 20 classes and skips some small instances); "classification" errors between cat/dog and sofa/chair look like genuinely confusable classes, and sofa/couch label conventions differ between COCO and VOC. Checking either needs manual review of the logged examples.
