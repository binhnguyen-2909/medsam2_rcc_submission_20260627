"""
Đề xuất AUTO-BOX cho 1 ảnh: cắt thước+nhãn -> lưới 30 ô -> ensemble 100 model
biểu quyết đa số -> bbox cụm ô liền kề. Dùng cho app gán nhãn (prefill box).

  from propose_box import load_proposer, propose
  proposer = load_proposer()
  box = propose(proposer, bgr)        # (x0,y0,x1,y1) hoặc None
"""
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from specimen_clean import clean_specimen
from train_cellbox import (MLP, cell_features, grid_cells, auto_box)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_proposer(path=os.path.join(ROOT, "models/cellbox_ensemble.pt")):
    ck = torch.load(path, map_location=DEVICE)
    models = []
    for st in ck["states"]:
        m = MLP(ck["d"]).to(DEVICE); m.load_state_dict(st); m.eval()
        models.append(m)
    return {"models": models, "mu": ck["mu"], "sd": ck["sd"],
            "vote_thr": ck["vote_thr"], "prob_thr": ck["prob_thr"]}


def propose(proposer, bgr, return_dbg=False, multi=False):
    H, W = bgr.shape[:2]
    spec, bb, sdbg = clean_specimen(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    cells, cols, rows = grid_cells(*bb, landscape=(W >= H))
    rec = {"cols": cols, "rows": rows, "cells": cells}

    feats, valid_ci = [], []
    for ci, (cx0, cy0, cx1, cy1) in enumerate(cells):
        f, _ = cell_features(rgb, hsv, lab, spec, cx0, cy0, cx1, cy1, bb, gray)
        if f is not None:
            feats.append(f); valid_ci.append(ci)
    if not feats:
        return (None, {"spec_bbox": bb}) if return_dbg else None

    X = (np.array(feats, np.float32) - proposer["mu"]) / proposer["sd"]
    Xt = torch.tensor(X, device=DEVICE)
    votes = np.zeros(len(feats), np.float32)
    with torch.no_grad():
        for m in proposer["models"]:
            p = torch.sigmoid(m(Xt)).cpu().numpy()
            votes += (p >= proposer["prob_thr"]).astype(np.float32)
    votes /= len(proposer["models"])
    vote_by_cell = {ci: float(v) for ci, v in zip(valid_ci, votes)}
    box = auto_box(rec, vote_by_cell, vote_thr=proposer["vote_thr"], multi=multi)
    if return_dbg:
        return box, {"spec_bbox": bb, "votes": vote_by_cell, "cells": cells,
                     "ruler": sdbg.get("ruler"), "label": sdbg.get("label")}
    return box


if __name__ == "__main__":
    import json
    P = load_proposer()
    done = json.load(open(os.path.join(ROOT, "labels/done.json")))
    test = json.load(open(os.path.join(ROOT, "labels/test_frozen.json")))["test"]
    os.makedirs(os.path.join(ROOT, "scratch_box"), exist_ok=True)
    paths = []
    for s in test[:6]:
        ip = os.path.join(ROOT, "data/20241212", s + ".jpg")
        bgr = cv2.imread(ip); H, W = bgr.shape[:2]
        box, dbg = propose(P, bgr, return_dbg=True)
        vis = bgr.copy()
        tp = os.path.join(ROOT, "labels/masks", s + ".png")
        if os.path.isfile(tp):
            tm = (cv2.imread(tp, 0) > 127).astype(np.uint8)
            cnts, _ = cv2.findContours(tm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 0, 255), 4)   # u thật: đỏ
        if box:
            cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 6)  # auto-box: green
        p = os.path.join(ROOT, "scratch_box", s[:12] + ".jpg")
        cv2.imwrite(p, cv2.resize(vis, (W // 3, H // 3))); paths.append(p)
    print("\n".join(paths))
