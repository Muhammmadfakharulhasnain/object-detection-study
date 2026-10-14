# Experiment 5 - input resolution

Setup: YOLOv8n with COCO weights (no training), 1,000 random Pascal VOC2007-test images (seed 0), 2422 non-difficult objects, 20 VOC classes, own NMS and own evaluator (VOC matching, IoU 0.5). Unless stated: 640 px input, clean images. Clean images letterboxed to 320 / 480 / 640 / 800 px. Latency: one image at a time, 1 CPU thread, this machine only (`scripts/measure_latency.py`).

|   input_px |   precision |   recall |    f1 |   mAP50 |   mAP50_95 |   small_recall |   forward_ms |   pipeline_ms |
|-----------:|------------:|---------:|------:|--------:|-----------:|---------------:|-------------:|--------------:|
|        320 |       0.762 |    0.75  | 0.756 |   0.792 |      0.575 |          0.33  |       55.15  |        63.902 |
|        480 |       0.739 |    0.828 | 0.781 |   0.84  |      0.624 |          0.602 |      100.114 |       105.525 |
|        640 |       0.71  |    0.837 | 0.769 |   0.827 |      0.604 |          0.757 |      178.125 |       186.464 |
|        800 |       0.679 |    0.827 | 0.746 |   0.802 |      0.551 |          0.777 |      266.974 |       279.077 |

**Result.** mAP@50 is highest at **480 px** (0.840), above 640 px (0.827) and 800 px (0.802); mAP@50:95 also peaks at 480 px (0.624). Recall on small objects keeps rising (0.33 -> 0.60 -> 0.76 -> 0.78) while precision falls (0.76 -> 0.68). Compute grows about 4.4x from 320 to 800 px.

**Answer.** Going beyond 480 px did not pay off on this subset: it costs 1.8x time at 640 px for slightly lower mAP. Larger inputs help small objects but add false positives.

**Caution.** A peak below the training resolution (YOLOv8 was trained at 640) is not what one would first expect. A possible reason is that most VOC objects are large (56% of objects here), so upscaling helps few of them, but this is a hypothesis; it was not tested (for example by evaluating small objects only at each resolution with more data).
