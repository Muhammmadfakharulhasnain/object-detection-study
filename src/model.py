"""Thin wrapper around a pretrained YOLOv8 network.

The framework is only used for the forward pass. Preprocessing (letterbox), decoding of the raw
output, confidence filtering and NMS are all done with this repo's own code, so every
post-processing step can be inspected and varied.
"""
import time
import cv2
import numpy as np
import torch
from .dataset import VOC_TO_COCO_IDX, VOC_TO_COCO_NAME
from .boxes import cxcywh_to_xyxy, clip_boxes


def load_yolo(weights="yolov8n.pt"):
    from ultralytics import YOLO
    y = YOLO(weights)
    net = y.model.float().eval()
    names = [y.names[i] for i in VOC_TO_COCO_IDX]
    assert names == VOC_TO_COCO_NAME, "VOC -> COCO class mapping does not match the model's class names"
    return net, y


def letterbox(img_bgr, size, pad_value=114):
    """Resize keeping aspect ratio, pad to size x size. Returns (rgb_image, scale, (pad_x, pad_y))."""
    h, w = img_bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), r, (left, top)


def to_tensor(rgb_images):
    arr = np.stack(rgb_images).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr))


@torch.inference_mode()
def forward_raw(net, batch_tensor):
    """Raw network output, shape (B, 4 + 80, N): box (cx, cy, w, h in input pixels) + 80 class scores."""
    out = net(batch_tensor)
    return (out[0] if isinstance(out, (list, tuple)) else out).cpu().numpy()


def decode_candidates(pred, scale, pad, orig_hw, min_conf=0.01):
    """Raw output of ONE image -> candidate detections in original-image pixels.

    Multi-label: an anchor can produce one candidate for every VOC class whose score >= min_conf.
    Only the 20 VOC categories are kept (COCO-only classes are discarded)."""
    cls_scores = pred[4:][VOC_TO_COCO_IDX]                       # (20, N)
    boxes_cxcywh = pred[:4].T                                    # (N, 4)
    cls_i, anc_i = np.where(cls_scores >= min_conf)
    scores = cls_scores[cls_i, anc_i]
    boxes = cxcywh_to_xyxy(boxes_cxcywh[anc_i])
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes /= scale
    boxes = clip_boxes(boxes, orig_hw[1], orig_hw[0])
    ok = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
    return {"boxes": boxes[ok].astype(np.float32), "scores": scores[ok].astype(np.float32),
            "labels": cls_i[ok].astype(np.int16)}


def decode_all80_argmax(pred, conf_thr):
    """Ultralytics 'predict' style decoding (for the equivalence check): each anchor keeps only its
    best of the 80 COCO classes. Returns boxes (in input-tensor pixels), scores, coco class ids."""
    scores_all = pred[4:]                                        # (80, N)
    cls = scores_all.argmax(0)
    sc = scores_all.max(0)
    m = sc >= conf_thr
    return cxcywh_to_xyxy(pred[:4].T[m]), sc[m], cls[m]


def run_candidates(net, image_paths, size, transform=None, min_conf=0.01, batch=8):
    """Run the detector over images. Returns {image_id: candidates} and mean ms/image (batched)."""
    ids = list(image_paths)
    result, total = {}, 0.0
    for s in range(0, len(ids), batch):
        chunk = ids[s:s + batch]
        rgbs, metas = [], []
        for i in chunk:
            im = cv2.imread(image_paths[i])
            if transform is not None:
                im = transform(im)
            rgb, r, pad = letterbox(im, size)
            rgbs.append(rgb); metas.append((r, pad, im.shape[:2]))
        x = to_tensor(rgbs)
        t0 = time.perf_counter()
        pred = forward_raw(net, x)
        total += time.perf_counter() - t0
        for k, i in enumerate(chunk):
            r, pad, hw = metas[k]
            result[i] = decode_candidates(pred[k], r, pad, hw, min_conf)
    return result, 1000 * total / len(ids)
