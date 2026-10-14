# Experiment 3 - object size

Setup: YOLOv8n with COCO weights (no training), 1,000 random Pascal VOC2007-test images (seed 0), 2422 non-difficult objects, 20 VOC classes, own NMS and own evaluator (VOC matching, IoU 0.5). Unless stated: 640 px input, clean images. Confidence 0.25, NMS 0.5. Sizes by ground-truth box area in the original image: small < 32^2 px, medium 32^2 - 96^2, large >= 96^2. Objects and unmatched predictions outside the group are ignored.

| size   |   n_gt |   precision |   recall |   ap50_at_conf0_01 |
|:-------|-------:|------------:|---------:|-------------------:|
| small  |    103 |       0.241 |    0.757 |              0.615 |
| medium |    758 |       0.676 |    0.802 |              0.744 |
| large  |   1561 |       0.823 |    0.859 |              0.864 |
| all    |   2422 |       0.71  |    0.837 |              0.827 |

**Result.** Recall 0.757 / 0.802 / 0.859 and AP@50 0.615 / 0.744 / 0.864 for small / medium / large. Performance does fall with object size, as expected.

**Cautions.** Only 103 small objects exist in this subset (4.3%), so the small-object numbers are noisy. The very low small-object precision (0.241) mixes real false alarms with tiny predicted boxes on things that VOC did not annotate; this experiment cannot separate the two.
