# Experiment 1 - confidence threshold

Setup: YOLOv8n with COCO weights (no training), 1,000 random Pascal VOC2007-test images (seed 0), 2422 non-difficult objects, 20 VOC classes, own NMS and own evaluator (VOC matching, IoU 0.5). Unless stated: 640 px input, clean images. NMS IoU fixed at 0.5; only the confidence threshold changes.

**Question.** How does raising the confidence threshold change the number of detections, precision and recall?

|   conf |   n_det |   tp |   fp |   fn |   dets_per_img |   precision |   recall |    f1 |   map50_truncated |
|-------:|--------:|-----:|-----:|-----:|---------------:|------------:|---------:|------:|------------------:|
|   0.1  |    4330 | 2176 | 2154 |  246 |          4.695 |       0.503 |    0.898 | 0.645 |             0.803 |
|   0.25 |    2853 | 2027 |  826 |  395 |          3.132 |       0.71  |    0.837 | 0.769 |             0.764 |
|   0.5  |    2017 | 1772 |  245 |  650 |          2.177 |       0.879 |    0.732 | 0.798 |             0.679 |
|   0.75 |    1233 | 1190 |   43 | 1232 |          1.278 |       0.965 |    0.491 | 0.651 |             0.473 |

(`n_det` counts matched TP + FP; `dets_per_img` counts every kept box. `map50_truncated` is AP computed only from boxes above the threshold.)

**Result.** From 0.10 to 0.75 precision rises 0.503 -> 0.965, recall falls 0.898 -> 0.491, and the detector returns 4.70 -> 1.28 boxes per image (true objects per image: 2.42). On the finer 0.05 grid F1 peaks at confidence 0.50 (F1 0.798); the common default 0.25 gives F1 0.769. mAP@50 computed from the truncated list falls 0.803 -> 0.473, because removing low-confidence boxes caps the recall the curve can reach.

**Reading.** The threshold is a choice of operating point, not a quality setting: 0.10 finds more objects but about half its boxes are wrong; 0.75 is almost always right but misses half the objects. mAP is therefore computed from a very low threshold (here 0.01) so that it does not depend on this choice.

**Limits.** One detector, one subset, one NMS setting. Whether 0.50 is best depends on how much a false alarm costs compared with a miss; F1 weighs them equally.
