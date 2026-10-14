# Final detection study - summary of findings

Setup: YOLOv8n with COCO weights (no training), 1,000 random Pascal VOC2007-test images (seed 0), 2422 non-difficult objects, 20 VOC classes, own NMS and own evaluator (VOC matching, IoU 0.5). Unless stated: 640 px input, clean images.

**Baseline** (confidence 0.25, NMS 0.5, 640 px): precision 0.710, recall 0.837, F1 0.769; mAP@50 0.827, mAP@50:95 0.604.

**Main sensitivity table** (confidence 0.25, NMS 0.5; mAP from the full ranked list):

| condition         |   precision |   recall |   map50 |   map50_95 |
|:------------------|------------:|---------:|--------:|-----------:|
| clean             |       0.71  |    0.837 |   0.827 |      0.604 |
| blur_sigma3       |       0.727 |    0.543 |   0.561 |      0.365 |
| dark_x0.5         |       0.724 |    0.833 |   0.823 |      0.6   |
| bright_x1.6       |       0.704 |    0.809 |   0.803 |      0.576 |
| low_contrast_x0.4 |       0.733 |    0.784 |   0.791 |      0.565 |
| noise_sigma25     |       0.749 |    0.599 |   0.611 |      0.406 |
| jpeg_q10          |       0.603 |    0.29  |   0.287 |      0.167 |

**What the experiments say**
1. Confidence threshold: pure precision/recall trade-off; best F1 at 0.50, not at the usual 0.25 (experiment 1).
2. NMS: required; 0.5 is best for mAP; the low end (0.3) costs recall in crowded scenes, the high end (0.7) leaves duplicates (experiment 2).
3. Size: small objects are harder, but only 103 exist in the subset (experiment 3).
4. Image conditions: JPEG q10, blur and noise at the tested strengths hurt a lot; darkening, brightening, lower contrast little. The best confidence threshold drops for degraded inputs, the best NMS threshold hardly changes (experiment 4).
5. Resolution: best at 480 px on this subset, with 640/800 px costing more time (experiment 5).
6. Errors: after NMS, remaining errors are mostly mislabelled/poorly placed boxes, boxes on unannotated things, and correct boxes with low scores (experiment 6).

**Validation of the from-scratch code**
* NMS identical to `torchvision.ops.batched_nms` on 1000/1000 images at each of IoU 0.3, 0.5, 0.7.
* Decode + filter + NMS identical to Ultralytics' own `predict()` on 60/60 images (max box difference 0.10 px).
* Evaluator vs pycocotools on the same detections: mAP@50 0.8271 (ours) vs 0.8270 (COCO tool); mAP@50:95 0.6041 vs 0.6058. Small differences are expected (COCO uses 101-point interpolation and slightly different matching).
* 20 unit tests (`pytest`).

**Limitations**
* No detector was trained: a COCO-pretrained YOLOv8n is evaluated zero-shot on VOC classes. Label-convention differences (sofa/couch, diningtable) cost some AP.
* One detector, one 1,000-image subset of VOC2007 test (about 20% of it), one severity per corruption, one noise seed, no confidence intervals. Differences of a few mAP points (for example 640 vs 800 px) should not be over-read.
* COCO and VOC images both come from Flickr; overlap between COCO training images and VOC2007 test images cannot be excluded.
* Latency was measured on a single CPU thread of one machine.

**Future work:** several corruption severities and noise seeds; bootstrap confidence intervals; compare a second detector (e.g. a two-stage model); fine-tune on VOC train; check the "background" false positives by hand.
