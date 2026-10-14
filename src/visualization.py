"""Plot helpers (matplotlib)."""
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from .dataset import VOC_CLASSES


def draw_boxes(ax, boxes, labels=None, scores=None, color="lime", lw=2, linestyle="-", fontsize=8):
    for i, b in enumerate(np.asarray(boxes).reshape(-1, 4)):
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                       edgecolor=color, linewidth=lw, linestyle=linestyle))
        txt = ""
        if labels is not None:
            txt += VOC_CLASSES[int(labels[i])] if not isinstance(labels[i], str) else labels[i]
        if scores is not None:
            txt += f" {scores[i]:.2f}"
        if txt:
            ax.text(b[0], max(b[1] - 3, 8), txt, color="black", fontsize=fontsize,
                    bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor="none"))


def show_image(ax, path_or_array):
    img = cv2.imread(path_or_array) if isinstance(path_or_array, str) else path_or_array
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.axis("off")


def gt_vs_pred(ax, path, gt, det, title=None):
    """Green = ground truth, red = prediction."""
    show_image(ax, path)
    draw_boxes(ax, gt["boxes"], gt["labels"], color="lime", lw=2)
    draw_boxes(ax, det["boxes"], det["labels"], det["scores"], color="red", lw=1.5, linestyle="--")
    if title:
        ax.set_title(title, fontsize=9)
