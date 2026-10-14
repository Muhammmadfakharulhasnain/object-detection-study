"""Independent check that the JPEG-quality drop is real: count detections from Ultralytics' own predict()
(not our pipeline) on the same 40 images at several JPEG qualities. Writes results/jpeg_sanity_check.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2, torch
torch.set_num_threads(1)
from src.dataset import load_voc_subset
from src.corruptions import jpeg
from src.model import load_yolo
from src.experiments import load_gts
ids, _ = load_gts(); _, _, paths = load_voc_subset(sys.argv[1], 1000, 0)
_, yolo = load_yolo("yolov8n.pt")
n = {"clean": 0, 10: 0, 30: 0, 75: 0}; psnr = []; size_ratio = []
for i in ids[:40]:
    im = cv2.imread(paths[i])
    for q in n:
        x = im if q == "clean" else jpeg(im, q)
        if q == 10:
            psnr.append(cv2.PSNR(im, x)); size_ratio.append(len(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 10])[1]) / os.path.getsize(paths[i]))
        n[q] += len(yolo.predict(x, conf=0.25, iou=0.5, verbose=False, device="cpu", imgsz=640)[0].boxes)
out = {"images": 40, "detections_ultralytics_predict": {str(k): v for k, v in n.items()}, "psnr_q10_db": float(np.mean(psnr)), "q10_file_size_vs_original": float(np.mean(size_ratio))}
json.dump(out, open("results/jpeg_sanity_check.json", "w"), indent=1); print(out)
