"""
Dựng lại labels/annotations.csv từ NGUỒN ĐÚNG = file mask + prompts + Excel.
Khắc phục lệch schema giữa app cũ/mới. Schema mới thống nhất.
  python rebuild_csv.py
"""
import csv
import json
import os
from glob import glob

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MASKS = os.path.join(ROOT, "labels/masks")
PROMPTS = os.path.join(ROOT, "labels/prompts")
IMG = os.path.join(ROOT, "data/20241212")
EXCEL_CSV = os.path.join(ROOT, "processed/excel_parsed.csv")
OUT = os.path.join(ROOT, "labels/annotations.csv")

EXCEL = {}
if os.path.isfile(EXCEL_CSV):
    for r in csv.DictReader(open(EXCEL_CSV)):
        EXCEL[r["canon"]] = r

rows = []
for mp in sorted(glob(os.path.join(MASKS, "*.png"))):
    stem = os.path.splitext(os.path.basename(mp))[0]
    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if m is None:
        continue
    union = int((m > 127).sum())
    H, W = m.shape[:2]
    n_obj, last_box, last_score = "", "", ""
    pj = os.path.join(PROMPTS, stem + ".json")
    if os.path.isfile(pj):
        try:
            d = json.load(open(pj))
            n_obj = d.get("n_objects", "")
            if "instances" in d and d["instances"]:
                last = d["instances"][-1]
                last_box = "|".join(str(int(v)) for v in last["box"]) if last.get("box") else ""
                last_score = last.get("score", "")
            else:  # schema app cũ
                last_box = "|".join(str(int(v)) for v in d["box"]) if d.get("box") else ""
                last_score = d.get("score", "")
                n_obj = n_obj or 1
        except Exception:
            pass
    pid = stem.split("^")[0]
    er = EXCEL.get(pid.replace("-", ""), {})
    rows.append({"stem": stem, "patient_id": pid, "n_objects": n_obj,
                 "union_area_px": union, "last_box": last_box,
                 "last_score": round(last_score, 4) if isinstance(last_score, float) else last_score,
                 "W": W, "H": H, "mass_dims_cm": er.get("mass_dims", ""),
                 "mass_area_cm2": er.get("mass_area_cm2", "")})

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"Dựng lại {len(rows)} dòng -> {OUT}")
print(f"bệnh nhân: {len(set(r['patient_id'] for r in rows))}")
