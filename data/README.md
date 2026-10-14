# Data

Nothing is stored here. Experiments use **Pascal VOC 2007 test** (4,952 images, 20 classes).

```bash
curl -L -o VOCtest_06-Nov-2007.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtest_06-Nov-2007.zip
unzip VOCtest_06-Nov-2007.zip -d data/        # creates data/VOCdevkit/VOC2007/
curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
```

The study uses a random subset of 1,000 test images (`numpy.random.RandomState(0)` over `ImageSets/Main/test.txt`, see `src/dataset.py`). Image IDs and parsed annotations are stored in `results/cache/gts.pkl`.

| item | value |
|---|---|
| dataset / version | Pascal VOC 2007, test split |
| images used | 1,000 (seed 0) |
| objects | 3,004 annotated, 582 flagged *difficult* (ignored), 2,422 evaluated |
| classes | 20 (per-class counts: `results/dataset_summary.csv`) |
| train / validation split | none - no model is trained or tuned |
| preprocessing | letterbox resize to 320/480/640/800 px square (aspect ratio kept, grey padding), RGB, 0-1 scale |
