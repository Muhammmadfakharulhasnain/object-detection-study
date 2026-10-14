# Experiment 4 - image conditions

Setup: YOLOv8n with COCO weights (no training), 1,000 random Pascal VOC2007-test images (seed 0), 2422 non-difficult objects, 20 VOC classes, own NMS and own evaluator (VOC matching, IoU 0.5). Unless stated: 640 px input, clean images. Confidence 0.25, NMS 0.5 for precision/recall; mAP from the full ranked list. Corruptions leave geometry unchanged, so ground truth is reused. Parameters (`src/corruptions.py`): Gaussian blur sigma 3 px; brightness x0.5 and x1.6; contrast x0.4; Gaussian noise sigma 25 (seeded); JPEG quality 10. One severity each.

| condition         |   precision |   recall |    f1 |   dets_per_img |   map50 |   map50_95 |   d_map50 |
|:------------------|------------:|---------:|------:|---------------:|--------:|-----------:|----------:|
| clean             |       0.71  |    0.837 | 0.769 |          3.132 |   0.827 |      0.604 |     0     |
| blur_sigma3       |       0.727 |    0.543 | 0.622 |          1.871 |   0.561 |      0.365 |    -0.266 |
| dark_x0.5         |       0.724 |    0.833 | 0.775 |          3.045 |   0.823 |      0.6   |    -0.004 |
| bright_x1.6       |       0.704 |    0.809 | 0.753 |          3.036 |   0.803 |      0.576 |    -0.024 |
| low_contrast_x0.4 |       0.733 |    0.784 | 0.758 |          2.815 |   0.791 |      0.565 |    -0.036 |
| noise_sigma25     |       0.749 |    0.599 | 0.666 |          2.039 |   0.611 |      0.406 |    -0.216 |
| jpeg_q10          |       0.603 |    0.29  | 0.391 |          1.198 |   0.287 |      0.167 |    -0.54  |

**Result.** Change in mAP@50 against clean (0.827): blur_sigma3 -0.266, dark_x0.5 -0.004, bright_x1.6 -0.024, low_contrast_x0.4 -0.036, noise_sigma25 -0.216, jpeg_q10 -0.540. Darkening costs almost nothing here; blur and noise remove about a quarter of mAP@50; JPEG quality 10 removes most of it.
Precision barely moves (it even rises slightly for blur, noise and darkening) while recall drops: under degraded input the detector produces fewer boxes rather than more wrong ones.

**JPEG check.** The JPEG result is large enough that it was checked independently of this repo's pipeline: Ultralytics' own `predict()` returns 217 detections on 40 clean images, 206 at JPEG quality 75, 174 at 30 and 85 at 10 (quality-10 files are 10% of the original size, PSNR 25.3 dB). So the drop is real, not a bug in the corruption code.

## Do the best thresholds depend on the condition?

| condition         |   F1@conf0.1 |   F1@conf0.25 |   F1@conf0.5 |   F1@conf0.75 |   best_conf |   best_F1 |   F1@nms0.3 |   F1@nms0.5 |   F1@nms0.7 |
|:------------------|-------------:|--------------:|-------------:|--------------:|------------:|----------:|------------:|------------:|------------:|
| clean             |        0.645 |         0.769 |        0.798 |         0.651 |        0.5  |     0.798 |       0.769 |       0.769 |       0.747 |
| blur_sigma3       |        0.57  |         0.622 |        0.543 |         0.327 |        0.2  |     0.623 |       0.622 |       0.622 |       0.611 |
| dark_x0.5         |        0.651 |         0.775 |        0.79  |         0.641 |        0.4  |     0.799 |       0.776 |       0.775 |       0.755 |
| bright_x1.6       |        0.631 |         0.753 |        0.774 |         0.612 |        0.45 |     0.778 |       0.752 |       0.753 |       0.731 |
| low_contrast_x0.4 |        0.663 |         0.758 |        0.763 |         0.585 |        0.35 |     0.777 |       0.758 |       0.758 |       0.738 |
| noise_sigma25     |        0.6   |         0.666 |        0.6   |         0.377 |        0.25 |     0.666 |       0.663 |       0.666 |       0.653 |
| jpeg_q10          |        0.375 |         0.391 |        0.296 |         0.124 |        0.2  |     0.398 |       0.39  |       0.391 |       0.386 |

* **Confidence:** the best threshold is 0.50 on clean images but 0.20 (blur), 0.25 (noise) and 0.20 (JPEG): degraded images lower the scores of correct boxes, so a threshold tuned on clean data throws away detections. Using 0.50 on blur gives F1 0.543; the best threshold gives 0.623.
* **NMS IoU:** within 0.3-0.7 the F1 of every condition changes by at most 0.023; it is much less sensitive than confidence.

**Limits.** One severity per corruption and one noise seed, so this shows how much *these settings* hurt, not a ranking of corruption types. Explanations such as "blur removes the texture the detector uses" are hypotheses; they would need several severities and detectors to test.
