"""Pascal VOC loading. Boxes are returned as float xyxy in pixel coordinates."""
import os
import xml.etree.ElementTree as ET
import numpy as np

VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]

# Index of the same category in the 80-class COCO label list used by the pretrained YOLO model.
VOC_TO_COCO_IDX = [4, 1, 14, 8, 39, 5, 2, 15, 56, 19, 60, 16, 17, 3, 0, 58, 18, 57, 6, 62]
VOC_TO_COCO_NAME = ["airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                    "dining table", "dog", "horse", "motorcycle", "person", "potted plant", "sheep",
                    "couch", "train", "tv"]


def parse_voc_xml(path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    width, height = int(size.find("width").text), int(size.find("height").text)
    boxes, labels, difficult, truncated = [], [], [], []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        # VOC pixel indices are 1-based and inclusive -> convert to continuous coordinates
        boxes.append([x1 - 1, y1 - 1, x2, y2])
        labels.append(VOC_CLASSES.index(obj.find("name").text.strip()))
        d = obj.find("difficult")
        difficult.append(bool(int(d.text)) if d is not None else False)
        t = obj.find("truncated")
        truncated.append(bool(int(t.text)) if t is not None else False)
    return {"width": width, "height": height,
            "boxes": np.array(boxes, dtype=float).reshape(-1, 4),
            "labels": np.array(labels, dtype=int),
            "difficult": np.array(difficult, dtype=bool),
            "truncated": np.array(truncated, dtype=bool)}


def load_voc_subset(voc_root, n_images=1000, seed=0, split="test"):
    """Randomly sample n_images from a VOC split (fixed seed -> reproducible).

    voc_root is the VOC2007 folder (contains JPEGImages, Annotations, ImageSets).
    Returns (image_ids, gts, image_paths)."""
    with open(os.path.join(voc_root, "ImageSets", "Main", f"{split}.txt")) as f:
        all_ids = [l.strip() for l in f if l.strip()]
    rng = np.random.RandomState(seed)
    ids = sorted(rng.choice(all_ids, size=min(n_images, len(all_ids)), replace=False).tolist())
    gts, paths = {}, {}
    for i in ids:
        gts[i] = parse_voc_xml(os.path.join(voc_root, "Annotations", f"{i}.xml"))
        paths[i] = os.path.join(voc_root, "JPEGImages", f"{i}.jpg")
    return ids, gts, paths
