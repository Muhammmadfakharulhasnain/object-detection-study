from common import *
c = [md('''# 08 - Error analysis

**Question (guide, experiment 5):** where does the detector fail, and what kinds of failures are they? Every false positive and every missed object at the default operating point (confidence 0.25, NMS 0.5, IoU 0.5, 640 px, clean images) is sorted into a cause by `src/analysis.py` (rules listed in its docstring). Causes found here are *categories of what the box geometry shows*; why the detector behaves that way remains a hypothesis.'''),
code(SETUP + '''
import cv2
from src.experiments import *
from src.analysis import analyse_errors, FP_TYPES, FN_TYPES, size_bucket
from src.dataset import load_voc_subset, VOC_CLASSES
from src.visualization import show_image, draw_boxes
from src.evaluation import Evaluator
ids, gts = load_gts(); _, _, paths = load_voc_subset(VOC, 1000, 0)
cands = load_candidates("clean_640"); dets = detections(cands, 0.25, 0.5)
fps, fns = analyse_errors(gts, dets, cands, conf_thr=0.25)
fp, fn = pd.DataFrame(fps), pd.DataFrame(fns)
m = Evaluator(gts, dets).metrics(0.5)
print(f"TP {m['tp']}  FP {m['fp']} (taxonomy rows: {len(fp)})  FN {m['fn']} (taxonomy rows: {len(fn)})"); assert len(fp) == m["fp"] and len(fn) == m["fn"]'''),
md("## 1. False positives by type"),
code('''t1 = fp.kind.value_counts().reindex(FP_TYPES).fillna(0).astype(int).to_frame("count"); t1["share"] = t1["count"] / t1["count"].sum(); display(t1.round(3))
t1b = fp.groupby(["kind", "size"]).size().unstack(fill_value=0); display(t1b)
cl = fp[fp.kind == "classification"]; conf = cl.groupby(["cls", "gt_cls"]).size().sort_values(ascending=False).head(8).reset_index(name="count")
conf["detected as"] = conf.cls.map(lambda c_: VOC_CLASSES[c_]); conf["ground truth"] = conf.gt_cls.map(lambda c_: VOC_CLASSES[c_]); display(conf[["detected as", "ground truth", "count"]])'''),
md("## 2. Missed objects (false negatives) by cause, size and truncation"),
code('''t2 = fn.cause.value_counts().reindex(FN_TYPES).fillna(0).astype(int).to_frame("count"); t2["share"] = t2["count"] / t2["count"].sum(); display(t2.round(3))
allg = pd.DataFrame([dict(size=size_bucket(float((b[2]-b[0])*(b[3]-b[1]))), truncated=bool(t), cls=int(l), img=i, n_obj=len(g["labels"]))
                     for i, g in gts.items() for b, l, d, t in zip(g["boxes"], g["labels"], g["difficult"], g["truncated"]) if not d])
rate = lambda col: (fn.groupby(col).size() / allg.groupby(col).size()).fillna(0)
by_size = pd.DataFrame({"objects": allg.groupby("size").size(), "missed": fn.groupby("size").size(), "miss_rate": rate("size")}).reindex(["small", "medium", "large"]); display(by_size.round(3))
by_trunc = pd.DataFrame({"objects": allg.groupby("truncated").size(), "missed": fn.groupby("truncated").size(), "miss_rate": rate("truncated")}); display(by_trunc.round(3))
allg["crowd"] = pd.cut(allg.n_obj, [0, 1, 3, 100], labels=["1 object", "2-3 objects", "4+ objects"]); fn2 = fn.merge(allg[["img", "n_obj"]].drop_duplicates("img"), on="img"); fn2["crowd"] = pd.cut(fn2.n_obj, [0, 1, 3, 100], labels=["1 object", "2-3 objects", "4+ objects"])
by_crowd = pd.DataFrame({"objects": allg.groupby("crowd", observed=True).size(), "missed": fn2.groupby("crowd", observed=True).size()}); by_crowd["miss_rate"] = by_crowd.missed / by_crowd.objects; display(by_crowd.round(3))
by_cls = pd.DataFrame({"objects": allg.groupby("cls").size(), "missed": fn.groupby("cls").size()}).fillna(0); by_cls["miss_rate"] = by_cls.missed / by_cls.objects; by_cls.index = [VOC_CLASSES[i] for i in by_cls.index]
display(by_cls.sort_values("miss_rate", ascending=False).head(6).round(3))'''),
md("## 3. Does the mix of errors change with the image condition?"),
code('''names = [n for n in ["clean_640", "blur_sigma3_640", "dark_x0.5_640", "bright_x1.6_640", "low_contrast_x0.4_640", "noise_sigma25_640", "jpeg_q10_640"] if os.path.exists(f"{CACHE}/{n}.pkl")]
rows = []
for n in names:
    cd = load_candidates(n); dd = detections(cd, 0.25, 0.5); f_, n_ = analyse_errors(gts, dd, cd, conf_thr=0.25)
    r = dict(condition=n.replace("_640", "")); r.update({f"FP_{k}": sum(x["kind"] == k for x in f_) for k in FP_TYPES}); r.update({f"FN_{k}": sum(x["cause"] == k for x in n_) for k in FN_TYPES}); rows.append(r)
mix = pd.DataFrame(rows); mix.to_csv(f"{RES}/error_mix_by_condition.csv", index=False); display(mix)'''),
md("## 4. Failure gallery\n\nGreen = ground truth, red dashed = detections (score shown), **yellow** = the object or box this example is about. Examples are chosen by rule (highest-score false positives, largest missed objects), not hand-picked for looking bad or good."),
code('''import shutil
shutil.rmtree(f"{RES}/failure_cases", ignore_errors=True); os.makedirs(f"{RES}/failure_cases")
def panel(ax, e, kind):
    i = e["img"]; g, d = gts[i], dets[i]; show_image(ax, paths[i])
    draw_boxes(ax, g["boxes"], g["labels"], color="lime", lw=2); draw_boxes(ax, d["boxes"], d["labels"], d["scores"], color="red", lw=1.5, linestyle="--")
    hb = d["boxes"][e["det"]] if "det" in e else g["boxes"][e["gt"]]
    draw_boxes(ax, [hb], color="gold", lw=4); ax.set_title(f"{kind}: {i}", fontsize=9)
cases = []
for k in FP_TYPES:
    sub = [e for e in fps if e["kind"] == k]; sub.sort(key=lambda e: -e["score"])
    if sub: cases.append(("FP " + k, sub[0]))
for k in FN_TYPES:
    sub = [e for e in fns if e["cause"] == k]; sub.sort(key=lambda e: -e["area"])
    if sub: cases.append(("FN " + k, sub[0]))
n = len(cases); cols = 3; rws = int(np.ceil(n / cols)); fig, axes = plt.subplots(rws, cols, figsize=(5.5 * cols, 4.2 * rws))
for ax in axes.ravel(): ax.axis("off")
for ax, (kind, e) in zip(axes.ravel(), cases):
    panel(ax, e, kind)
    f2, a2 = plt.subplots(figsize=(6, 5)); panel(a2, e, kind); f2.savefig(f"{RES}/failure_cases/{kind.replace(' ', '_')}_{e['img']}.png", dpi=90); plt.close(f2)
plt.tight_layout(); plt.savefig(f"{FIG}/failure_cases.png", dpi=90); plt.show()
log = pd.concat([fp.assign(type="FP_" + fp.kind)[["img", "type", "cls", "size", "score"]], fn.assign(type="FN_" + fn.cause, score=np.nan)[["img", "type", "cls", "size", "score"]]])
log["class"] = log.cls.map(lambda c_: VOC_CLASSES[c_]); log.drop(columns="cls").to_csv(f"{RES}/failure_cases/error_log.csv", index=False)
print(len(log), "errors logged ->", "results/failure_cases/error_log.csv")'''),
md("## 5. From category to hypothesis\n\nOnly the first column below is a measurement; the second is a hypothesis for a later experiment."),
code('''d_ = lambda x: x if x == x else 0
display(Markdown(f"""
| what the data shows | hypothesis (not tested here) |
|---|---|
| FP split: {', '.join(f'{k} {int(t1.loc[k, "count"])}' for k in FP_TYPES)} | duplicates are rare once NMS is applied; most FPs are objects with the wrong label or boxes that fit poorly, or true objects missing from the VOC annotation |
| FN split: {', '.join(f'{k} {int(t2.loc[k, "count"])}' for k in FN_TYPES)} | `low_confidence` misses would be recovered by a lower threshold at a precision cost; `missed` ones need a better model/resolution |
| miss rate small / medium / large: {by_size.miss_rate['small']:.3f} / {by_size.miss_rate['medium']:.3f} / {by_size.miss_rate['large']:.3f} | small objects have few pixels at 640 px input (see resolution experiment) |
| miss rate truncated / not truncated: {by_trunc.miss_rate[True]:.3f} / {by_trunc.miss_rate[False]:.3f} | partly out-of-frame objects lack context |
| miss rate 1 object / 2-3 / 4+ per image: {by_crowd.miss_rate['1 object']:.3f} / {by_crowd.miss_rate['2-3 objects']:.3f} / {by_crowd.miss_rate['4+ objects']:.3f} | crowding may increase misses through overlap and NMS (the pattern is not monotone here) |
"""))''')]
save(c, "08_error_analysis")
