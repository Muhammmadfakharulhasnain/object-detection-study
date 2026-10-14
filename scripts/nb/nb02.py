from common import *
c = [md('''# 02 - IoU from scratch

**Question:** how do we turn "these two boxes agree" into a single number between 0 and 1, and what does that number mean?'''),
code(SETUP + '''
import timeit
from src.boxes import box_area, intersection_box, intersection_area, union_area
from src.iou import calculate_iou, iou_matrix
from src.visualization import draw_boxes'''),
md('''## 1. Concept

```
                 area(A ∩ B)                 intersection
IoU(A, B) = -------------------------  =  ------------------
              area(A) + area(B) - area(A ∩ B)         union
```

* Intersection = the rectangle both boxes cover. Its corners are `max` of the left/top edges and `min` of the right/bottom edges. If the result has negative width or height the boxes do not overlap, so we clip to 0.
* Union = both areas added, minus the intersection (otherwise the shared part is counted twice).
* IoU is **not** "what fraction of the ground truth is covered". A small box fully inside a large one covers nothing of the large box's extra area, and IoU punishes that.'''),
md("## 2. Small example, step by step\n\nA = (0, 0, 10, 10), B = (5, 0, 15, 10): they share a 5 x 10 strip."),
code('''A, B = [0, 0, 10, 10], [5, 0, 15, 10]
print("intersection box :", intersection_box(A, B))
print("intersection area:", intersection_area(A, B))
print("area A, area B   :", box_area(A), box_area(B))
print("union area       :", union_area(A, B))
print("IoU              :", calculate_iou(A, B), "= 50 / 150")'''),
md("## 3. Implementation\n\nSee `src/iou.py`. `calculate_iou` mirrors the four steps above; `iou_matrix` does all pairs at once with NumPy broadcasting (needed for NMS and matching)."),
code('''import inspect; print(inspect.getsource(calculate_iou))'''),
md("## 4. Tests\n\nEvery case has a value that can be worked out by hand."),
code('''cases = [("identical", [0,0,10,10], [0,0,10,10], 1.0),
         ("disjoint", [0,0,10,10], [20,20,30,30], 0.0),
         ("touching edge only", [0,0,10,10], [10,0,20,10], 0.0),
         ("half shifted", [0,0,10,10], [5,0,15,10], 50/150),
         ("one inside the other", [0,0,10,10], [2,2,8,8], 36/100),
         ("zero-area boxes", [0,0,0,0], [0,0,0,0], 0.0)]
rows = []
for name, a, b, exp in cases:
    got = calculate_iou(a, b); assert abs(got - exp) < 1e-9 and abs(got - calculate_iou(b, a)) < 1e-12
    rows.append((name, round(exp, 4), round(got, 4), "ok"))
display(pd.DataFrame(rows, columns=["case", "expected (by hand)", "calculate_iou", "check"]))

rng = np.random.RandomState(0)
def rand_boxes(n):
    xy = rng.rand(n, 2) * 100; wh = rng.rand(n, 2) * 50 + 1; return np.concatenate([xy, xy + wh], 1)
a, b = rand_boxes(200), rand_boxes(150)
M = iou_matrix(a, b)
assert all(abs(M[i, j] - calculate_iou(a[i], b[j])) < 1e-9 for i in range(0, 200, 7) for j in range(0, 150, 7))
t_loop = timeit.timeit(lambda: [[calculate_iou(x, y) for y in b] for x in a], number=1)
t_vec = timeit.timeit(lambda: iou_matrix(a, b), number=1)
print(f"vectorised == scalar. 200 x 150 pairs: loop {t_loop*1000:.0f} ms, vectorised {t_vec*1000:.1f} ms")'''),
md("## 5. Visualisation: what do different IoU values look like?\n\nGround truth (green) and a prediction (red) slid sideways. The numbers are *computed*, not chosen."),
code('''gt = np.array([20, 20, 120, 120.])
shifts = [150, 90, 55, 25, 0]
fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
for ax, s in zip(axes, shifts):
    pred = gt + np.array([s, 0, s, 0])
    ax.set_xlim(0, 300); ax.set_ylim(150, 0); ax.set_aspect("equal")
    inter = intersection_box(gt, pred)
    if inter[2] > inter[0]:
        ax.add_patch(plt.Rectangle(inter[:2], inter[2]-inter[0], inter[3]-inter[1], color="gold", alpha=0.6))
    draw_boxes(ax, [gt], color="green"); draw_boxes(ax, [pred], color="red", linestyle="--")
    ax.set_title(f"shift {s}px  ->  IoU = {calculate_iou(gt, pred):.2f}")
plt.tight_layout(); plt.savefig(f"{FIG}/iou_examples.png", dpi=110); plt.show()

# IoU as a function of shift, and of size mismatch
sh = np.arange(0, 101, 2)
iou_shift = [calculate_iou(gt, gt + [s, 0, s, 0]) for s in sh]
sc = np.linspace(0.3, 2.0, 60)
iou_scale = [calculate_iou(gt, [70-50*k, 70-50*k, 70+50*k, 70+50*k]) for k in sc]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(sh, iou_shift); ax[0].set_xlabel("sideways shift (px), box is 100 px wide"); ax[0].set_ylabel("IoU")
for t in (0.5, 0.75): ax[0].axhline(t, ls=":", c="gray")
ax[1].plot(sc, iou_scale); ax[1].set_xlabel("prediction size / ground-truth size (same centre)"); ax[1].axhline(0.5, ls=":", c="gray")
plt.tight_layout(); plt.savefig(f"{FIG}/iou_vs_shift_and_scale.png", dpi=110); plt.show()
print("shift at which IoU first drops below 0.5:", sh[np.argmax(np.array(iou_shift) < 0.5)], "px of a 100 px box")
print("a box of half the area, fully inside the GT, has IoU =", calculate_iou(gt, [20, 20, 120, 70]))'''),
md("## 6. IoU on real detector output\n\nHow well do correct detections actually fit? For every ground-truth object matched by a detection (conf >= 0.25, NMS 0.5, YOLOv8n at 640 px), the IoU of the match:"),
code('''from src.experiments import load_gts, load_candidates, detections
ids, gts = load_gts(); cands = load_candidates("clean_640")
dets = detections(cands, 0.25, 0.5)
best = []
for i in ids:
    g, d = gts[i], dets[i]
    if not len(g["boxes"]): continue
    if not len(d["boxes"]):
        best.extend([0.0] * int((~g["difficult"]).sum())); continue          # nothing detected -> no match
    M = iou_matrix(d["boxes"], g["boxes"])
    M[d["labels"][:, None] != g["labels"][None, :]] = 0          # same class only
    M[:, g["difficult"]] = 0
    best.extend(M.max(0)[~g["difficult"]])                        # best same-class IoU per evaluated GT
best = np.array(best)
print(f"{len(best)} evaluated objects; fraction with a same-class detection at IoU>=0.5: {(best>=0.5).mean():.3f}, >=0.75: {(best>=0.75).mean():.3f}, >=0.9: {(best>=0.9).mean():.3f}")
fig, ax = plt.subplots(figsize=(6, 3.3)); ax.hist(best[best > 0], bins=40); ax.axvline(0.5, c="r", ls=":")
ax.set_xlabel("best same-class IoU per ground-truth object (zeros not shown)"); plt.tight_layout()
plt.savefig(f"{FIG}/iou_of_real_detections.png", dpi=110); plt.show()
pd.DataFrame({"IoU threshold": [0.5, 0.75, 0.9], "objects with a match": [(best>=t).mean() for t in (0.5,0.75,0.9)]}).to_csv(f"{RES}/iou_match_rates.csv", index=False)'''),
md('''## 7. Observation

* IoU is symmetric, between 0 and 1, and falls quickly with a small shift: in the example a shift of about a third of the box width already pulls IoU under 0.5 (section 5 prints the exact pixel value).
* A prediction that is entirely inside the ground truth but half its area scores only 0.5, which is why IoU is a strict measure and why `mAP@50:95` (which also demands IoU 0.75, 0.9...) is much lower than `mAP@50` later.
* On real detections most matched boxes sit well above 0.5, and the share falls as the required IoU increases (section 6). That gap between "found it" and "fitted it tightly" is what the IoU threshold in the metric controls.''')]
save(c, "02_iou_from_scratch")
