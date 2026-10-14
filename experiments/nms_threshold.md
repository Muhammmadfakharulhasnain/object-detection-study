# Experiment 2 - NMS IoU threshold

Setup: YOLOv8n with COCO weights (no training), 1,000 random Pascal VOC2007-test images (seed 0), 2422 non-difficult objects, 20 VOC classes, own NMS and own evaluator (VOC matching, IoU 0.5). Unless stated: 640 px input, clean images. Confidence fixed at 0.25; only the NMS IoU threshold changes ("off" = no NMS).

**Question.** What does NMS remove, and when does it become too aggressive?

| nms_iou   |   n_det |   tp |    fp |   fn |   dets_per_img |   precision |   recall |    f1 |   duplicate_fp |   crowded_person_recall |   map50_conf0_01 |
|:----------|--------:|-----:|------:|-----:|---------------:|------------:|---------:|------:|---------------:|------------------------:|-----------------:|
| 0.3       |    2758 | 1993 |   765 |  429 |          3.03  |       0.723 |    0.823 | 0.769 |              1 |                   0.856 |            0.811 |
| 0.5       |    2853 | 2027 |   826 |  395 |          3.132 |       0.71  |    0.837 | 0.769 |             13 |                   0.894 |            0.827 |
| 0.7       |    3033 | 2037 |   996 |  385 |          3.33  |       0.672 |    0.841 | 0.747 |            131 |                   0.899 |            0.822 |
| off       |   22981 | 2052 | 20929 |  370 |         25.019 |       0.089 |    0.847 | 0.162 |          16636 |                   0.913 |            0.211 |

`duplicate_fp` = false positives that sit on an object a better box already found. `crowded_person_recall` = recall for people in the 63 images with at least 4 people (367 people). `map50_conf0_01` = mAP@50 using the full ranked list.

**Result.**
* NMS is essential: without it there are 25.0 boxes per image and precision is 0.089 (16636 duplicate false positives).
* Threshold 0.7 leaves 131 duplicates vs 13 at 0.5 and 1 at 0.3, and F1 drops to 0.747 (0.3 and 0.5 give 0.769 and 0.769). mAP@50 is best at 0.5 (0.827) against 0.811 at 0.3 and 0.822 at 0.7.
* The cost of a low threshold shows up in crowded scenes: person recall there is 0.856 at 0.3, 0.894 at 0.5, 0.899 at 0.7. Over all images the difference is small (0.823 vs 0.841 recall).

**Answer to "when is NMS too aggressive?"** In this data only at the low end of the range, and mainly where objects of one class touch each other. Even then the effect is a few points of recall.

**Limits.** The crowded subset is 63 images, so the crowded-scene numbers are noisy; one detector; no soft-NMS or other variants were tried.
