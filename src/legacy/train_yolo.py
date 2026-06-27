"""
LOCALIZE #1 — YOLOv11 detector u (nhãn box thuần từ mask, vứt mask). Dataset YOLO từ
~1000 ảnh có nhãn (truth ưu tiên, else SAM), loại handdraw/e200/t12 patient. Train yolo11.
  python train_yolo.py --build --train --model yolo11s.pt --epochs 80
Lưu best -> checkpoints/yolo_best.pt. Env: medsam2_anno (cần ultralytics).
"""
import argparse, csv, json, os, sys, hashlib, shutil
import cv2, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from seg_crop import pat, frag_boxes
IMG_DIR = "data/20241212"; TRUTH_DIR = "labels_truth/masks"; SAM_DIR = "labels/masks"
YROOT = "yolo_ds"

def mask_path(s):
    tp = f"{TRUTH_DIR}/{s}.png"; return tp if os.path.isfile(tp) else f"{SAM_DIR}/{s}.png"

def build_dataset():
    hd = set(map(pat, json.load(open("labels_handdraw/select.json"))["stems"]))
    e200 = set(pat(r["stem"]) for r in csv.DictReader(open("results/confirm200_ft_vs_zs.csv")))
    t12 = set(map(pat, json.load(open("labels/test_frozen.json"))["test"]))
    excl = hd | e200 | t12
    truth = set(f[:-4] for f in os.listdir(TRUTH_DIR) if f.endswith(".png"))
    sam = set(f[:-4] for f in os.listdir(SAM_DIR) if f.endswith(".png"))
    stems = sorted((truth | sam))
    stems = [s for s in stems if pat(s) not in excl and os.path.isfile(f"{IMG_DIR}/{s}.jpg")]
    stems.sort(key=lambda s: hashlib.md5(s.encode()).hexdigest())
    nval = max(20, len(stems) // 10)
    val = set(stems[:nval]);
    for sp in ("train", "val"):
        os.makedirs(f"{YROOT}/images/{sp}", exist_ok=True); os.makedirs(f"{YROOT}/labels/{sp}", exist_ok=True)
    n = 0
    for s in stems:
        m = cv2.imread(mask_path(s), 0)
        if m is None: continue
        H, W = m.shape; boxes = frag_boxes(m > 127)
        if not boxes: continue
        sp = "val" if s in val else "train"
        ip = os.path.abspath(f"{IMG_DIR}/{s}.jpg"); dst = f"{YROOT}/images/{sp}/{s}.jpg"
        if not os.path.islink(dst) and not os.path.isfile(dst):
            try: os.symlink(ip, dst)
            except FileExistsError: pass
        with open(f"{YROOT}/labels/{sp}/{s}.txt", "w") as f:
            for x0, y0, x1, y1 in boxes:
                cx = (x0 + x1) / 2 / W; cy = (y0 + y1) / 2 / H; bw = (x1 - x0) / W; bh = (y1 - y0) / H
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        n += 1
    yaml = f"path: {os.path.abspath(YROOT)}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['tumor']\n"
    open(f"{YROOT}/data.yaml", "w").write(yaml)
    print(f"[yolo] dataset: {n} ảnh ({len(stems)-len(val)} train / {len(val)} val) -> {YROOT}", flush=True)

def train(model, epochs):
    from ultralytics import YOLO
    yolo = YOLO(model)
    yolo.train(data=f"{YROOT}/data.yaml", epochs=epochs, imgsz=1024, batch=4, device=0,
               project="yolo_runs", name="tumor", exist_ok=True, patience=20, verbose=False)
    best = "yolo_runs/tumor/weights/best.pt"
    if os.path.isfile(best):
        shutil.copy(best, "checkpoints/yolo_best.pt"); print(f"[yolo] best -> checkpoints/yolo_best.pt", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true"); ap.add_argument("--train", action="store_true")
    ap.add_argument("--model", default="yolo11s.pt"); ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()
    if args.build: build_dataset()
    if args.train: train(args.model, args.epochs)

if __name__ == "__main__":
    main()
