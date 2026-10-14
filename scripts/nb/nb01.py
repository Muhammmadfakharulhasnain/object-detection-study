from common import *
c = [md('''# 01 - Bounding boxes

**Question this notebook answers:** how is "where is the object?" written down as numbers, and what can go wrong with those numbers?

Plan: concept -> small example -> implementation (`src/boxes.py`) -> visualisation -> test -> observation.'''),
code(SETUP + '''
from src.boxes import *
from src.dataset import load_voc_subset, VOC_CLASSES
from src.visualization import show_image, draw_boxes'''),
md('''## 1. Concept

Classification answers *what* is in the image. Detection also answers *where*, and the usual answer is a rectangle.

Image coordinates start at the **top-left corner** `(0, 0)`. `x` grows to the right and `y` grows **downwards**. A rectangle can be described in two equivalent ways:

| format | numbers | used for |
|---|---|---|
| `xyxy` | `x_min, y_min, x_max, y_max` (left, top, right, bottom) | overlap maths, drawing, VOC annotations |
| `cxcywh` | `x_center, y_center, width, height` | what many detectors (including YOLO) predict internally |

They describe the same rectangle:

```
width    = x_max - x_min            x_center = (x_min + x_max) / 2
height   = y_max - y_min            y_center = (y_min + y_max) / 2
x_min    = x_center - width / 2     x_max    = x_center + width / 2
```

All code in this repo uses `xyxy` in pixels.'''),
md("## 2. Small example (by hand first)\n\nBox `(100, 50, 300, 250)`: width = 300-100 = 200, height = 250-50 = 200, center = (200, 150). The task from the guide: box `(100, 80, 300, 240)` -> width 200, height 160."),
code('''box = np.array([100, 50, 300, 250])
cxcywh = xyxy_to_cxcywh(box)
print("xyxy   :", box)
print("cxcywh :", cxcywh)
print("back   :", cxcywh_to_xyxy(cxcywh))
print("area   :", box_area(box))

# guide's first task
b = (100, 80, 300, 240)
print("task box width, height =", box_width(b), box_height(b))
assert (box_width(b), box_height(b)) == (200, 160)
assert np.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(box)), box)'''),
md("## 3. Visualisation on real images\n\nGround truth from Pascal VOC (2007 test). VOC stores pixel indices that start at 1 and include the last pixel, so the loader converts `xmin-1, ymin-1, xmax, ymax` into continuous coordinates."),
code('''ids, gts, paths = load_voc_subset(VOC, 1000, seed=0)
multi = [i for i in ids if 3 <= len(gts[i]["labels"]) <= 5][:3]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, i in zip(axes, multi):
    show_image(ax, paths[i]); g = gts[i]
    draw_boxes(ax, g["boxes"], g["labels"], color="lime")
    ax.set_title(f"{i}: " + ", ".join(VOC_CLASSES[l] for l in g["labels"]), fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIG}/bounding_box_examples.png", dpi=110); plt.show()'''),
md("## 4. What goes wrong when coordinates are wrong?\n\nFour typical mistakes, drawn on one object: swapped x/y, treating `cxcywh` as `xyxy`, forgetting to rescale after resizing, and normalised (0-1) coordinates used as pixels."),
code('''i = multi[0]; g = gts[i]; b = g["boxes"][0]; W, H = g["width"], g["height"]
wrong = {
  "correct": b,
  "x/y swapped": b[[1, 0, 3, 2]],
  "cxcywh read as xyxy": xyxy_to_cxcywh(b),
  "image resized x0.5, box not": b * 0.5 + 0 ,
  "normalised read as pixels (1px, invisible)": b / np.array([W, H, W, H]),
}
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, (name, bb) in zip(axes, wrong.items()):
    show_image(ax, paths[i]); draw_boxes(ax, [bb], color="lime" if name == "correct" else "red"); ax.set_title(name, fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIG}/bounding_box_mistakes.png", dpi=110); plt.show()
print("xyxy box:", b, " | area", box_area(b), "| degenerate boxes have area 0:", box_area([10, 10, 5, 5]))'''),
md("## 5. Tests"),
code('''rng = np.random.RandomState(0)
xy = rng.rand(1000, 2) * 100; wh = rng.rand(1000, 2) * 50 + 1
boxes = np.concatenate([xy, xy + wh], 1)
assert np.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(boxes)), boxes)       # round trip
assert np.allclose(box_area(boxes), wh[:, 0] * wh[:, 1])                  # area
assert box_area([5, 5, 5, 10]) == 0                                       # zero width
print("box tests passed (the full suite is in tests/test_core.py)")'''),
md("## 6. The dataset used in this project\n\nNumbers below are computed from the 1,000-image random subset (seed 0) of VOC2007 *test* used in every later experiment. Objects marked *difficult* are ignored by the evaluator (standard VOC protocol)."),
code('''rows = []
for c, name in enumerate(VOC_CLASSES):
    n = sum(int((g["labels"] == c).sum()) for g in gts.values())
    d = sum(int(((g["labels"] == c) & g["difficult"]).sum()) for g in gts.values())
    rows.append((name, n, d, n - d))
ds = pd.DataFrame(rows, columns=["class", "objects", "difficult", "evaluated"])
ds.loc[len(ds)] = ["TOTAL", ds.objects.sum(), ds.difficult.sum(), ds.evaluated.sum()]
ds.to_csv(f"{RES}/dataset_summary.csv", index=False); display(ds)
areas = np.concatenate([box_area(g["boxes"]) for g in gts.values()])
share = {"small (<32^2)": (areas < 32**2).mean(), "medium": ((areas >= 32**2) & (areas < 96**2)).mean(), "large (>=96^2)": (areas >= 96**2).mean()}
print("images:", len(ids), "| objects per image: mean %.2f, max %d" % (np.mean([len(g["labels"]) for g in gts.values()]), max(len(g["labels"]) for g in gts.values())))
print({k: round(v, 3) for k, v in share.items()})
fig, ax = plt.subplots(figsize=(6, 3)); ax.hist(np.sqrt(areas), bins=40); ax.set_xlabel("sqrt(box area) in pixels"); ax.set_ylabel("objects"); plt.tight_layout(); plt.savefig(f"{FIG}/box_size_distribution.png", dpi=110); plt.show()'''),
md('''## 7. Observation

* A box is just four numbers, but the *format* matters: every mistake in section 4 still produces a valid-looking rectangle, only in the wrong place. This is why box conversions get unit tests.
* The size distribution above matters later: objects are not equally easy to detect, and the experiments split results into small / medium / large using the usual COCO area limits (32^2 and 96^2 pixels).
* The evaluator only counts non-difficult objects ("evaluated" column).''')]
save(c, "01_bounding_boxes")
