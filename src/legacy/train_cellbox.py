"""
Auto-box bằng classifier mức Ô (cell).
Chia bbox bệnh phẩm thành lưới 30 ô (ngang 6x5 / dọc 5x6). Mỗi ô -> đặc trưng
màu+texture+vị trí; nhãn = tỉ lệ pixel U trong ô (từ 182 mask đã gán).
Train MLP nhỏ (torch) dự đoán P(ô = vết thương). Auto-box = bbox bao các ô P>thr.

Đánh giá CV theo BỆNH NHÂN (không rò rỉ):
  - AUC mức ô
  - IoU giữa auto-box và bbox-U-thật (gộp toàn ảnh)
  - recall: auto-box có chứa tâm khối u không

Chạy:  python train_cellbox.py            # CV + lưu model toàn data -> models/cellbox.pt
"""
import csv
import glob
import hashlib
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from specimen_clean import clean_specimen

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(ROOT, "data/20241212")
TUMOR_DIR = os.path.join(ROOT, "labels/masks")
SPEC_DIR = os.path.join(ROOT, "processed/mask")
META = os.path.join(ROOT, "processed/metadata.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRID = 30
POS_THR = 0.08          # ô coi là DƯƠNG nếu >=8% mô trong ô là u (u trải nhiều ô -> ô rìa cũng tính)
PROB_THR = 0.40         # 1 model coi ô dương nếu P>=PROB_THR
VOTE_THR = 0.50         # ô được CHỌN nếu >=50% trong 100 model bầu dương
ENSEMBLE = 100          # số lần train/chạy lại để biểu quyết đa số
MIN_SPEC_FRAC = 0.05    # ô có <5% mô thì coi là ngoài bệnh phẩm -> bỏ


def grid_cells(x0, y0, x1, y1, landscape):
    cols, rows = (6, 5) if landscape else (5, 6)
    xs = np.linspace(x0, x1 + 1, cols + 1).astype(int)
    ys = np.linspace(y0, y1 + 1, rows + 1).astype(int)
    cells = []
    for r in range(rows):
        for c in range(cols):
            cells.append((xs[c], ys[r], xs[c + 1], ys[r + 1]))
    return cells, cols, rows


def cell_features(rgb, hsv, lab, spec, cx0, cy0, cx1, cy1, bb, gray):
    sub_spec = spec[cy0:cy1, cx0:cx1]
    n = int(sub_spec.sum())
    area = max(1, (cy1 - cy0) * (cx1 - cx0))
    frac = n / area
    if frac < MIN_SPEC_FRAC:
        return None, frac
    m = sub_spec.astype(bool)
    feat = []
    for arr in (rgb, hsv, lab):
        sub = arr[cy0:cy1, cx0:cx1].reshape(-1, 3)[m.reshape(-1)]
        feat += list(sub.mean(0)) + list(sub.std(0))
    # texture: variance của Laplacian trong ô (chỉ vùng mô)
    lap = gray[cy0:cy1, cx0:cx1]
    feat.append(float(lap[m].var()))
    # vị trí tâm ô chuẩn hoá trong bbox bệnh phẩm
    bx0, by0, bx1, by1 = bb
    feat.append(((cx0 + cx1) / 2 - bx0) / max(1, bx1 - bx0))
    feat.append(((cy0 + cy1) / 2 - by0) / max(1, by1 - by0))
    return np.array(feat, dtype=np.float32), frac


def build_dataset(stems):
    X, Y, groups, idx = [], [], [], []   # idx: (stem, cell_index)
    per_img = {}
    for s in stems:
        ip = os.path.join(IMG_DIR, s + ".jpg")
        tp = os.path.join(TUMOR_DIR, s + ".png")
        if not (os.path.isfile(ip) and os.path.isfile(tp)):
            continue
        bgr = cv2.imread(ip)
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        gray = cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
        tumor = (cv2.imread(tp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        # specimen ĐÁNG TIN: đã cắt thước + nhãn (specimen_clean)
        spec, bb, _ = clean_specimen(bgr)
        cells, cols, rows = grid_cells(*bb, landscape=(W >= H))
        rec = {"cols": cols, "rows": rows, "cells": cells, "feat": [], "valid": [],
               "ratio": [], "tumor_px": [], "tumor_total": int(tumor.sum()),
               "spec_bbox": bb, "tumor_bbox": None}
        if tumor.sum() > 0:
            ys, xs = np.where(tumor)
            rec["tumor_bbox"] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        for ci, (cx0, cy0, cx1, cy1) in enumerate(cells):
            f, frac = cell_features(rgb, hsv, lab, spec, cx0, cy0, cx1, cy1, bb, gray)
            sub_t = tumor[cy0:cy1, cx0:cx1]
            sub_s = spec[cy0:cy1, cx0:cx1].astype(bool)
            denom = max(1, int(sub_s.sum()))
            ratio = float(sub_t[sub_s].sum()) / denom if f is not None else 0.0
            rec["feat"].append(f)
            rec["valid"].append(f is not None)
            rec["ratio"].append(ratio)
            rec["tumor_px"].append(int(sub_t.sum()))
            if f is not None:
                X.append(f); Y.append(1.0 if ratio >= POS_THR else 0.0)
                groups.append(s.split("^")[0]); idx.append((s, ci))
        per_img[s] = rec
    return (np.array(X, np.float32), np.array(Y, np.float32),
            np.array(groups), idx, per_img)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(X, Y, mu, sd, epochs=300, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    n = len(X)
    rng = np.random.default_rng(seed)
    bs = rng.integers(0, n, n)            # bootstrap mẫu -> mỗi lần khác nhau
    Xt = torch.tensor((X[bs] - mu) / sd, device=DEVICE)
    Yt = torch.tensor(Y[bs], device=DEVICE)
    pos_w = torch.tensor([(Y == 0).sum() / max(1, (Y == 1).sum())], device=DEVICE)
    m = MLP(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        loss = lossf(m(Xt), Yt); loss.backward(); opt.step()
    return m


def train_ensemble(X, Y, mu, sd, n=ENSEMBLE):
    """Train n model (seed + bootstrap khác nhau) = chạy đi chạy lại n lần."""
    return [train_model(X, Y, mu, sd, seed=k) for k in range(n)]


def predict_probs(m, X, mu, sd):
    with torch.no_grad():
        p = torch.sigmoid(m(torch.tensor((X - mu) / sd, device=DEVICE)))
    return p.cpu().numpy()


def ensemble_vote(models, X, mu, sd):
    """Trả về (vote_frac, mean_prob): tỉ lệ model bầu ô dương + xác suất TB."""
    Xn = torch.tensor((X - mu) / sd, device=DEVICE)
    votes = np.zeros(len(X), np.float32)
    psum = np.zeros(len(X), np.float32)
    with torch.no_grad():
        for m in models:
            p = torch.sigmoid(m(Xn)).cpu().numpy()
            votes += (p >= PROB_THR).astype(np.float32)
            psum += p
    return votes / len(models), psum / len(models)


def auc(y, p):
    pos = p[y == 1]; neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # AUC = P(score_pos > score_neg) qua xếp hạng
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def auto_box(rec, vote_by_cell, vote_thr=None, multi=False, min_cluster_cells=1):
    """Biểu quyết đa số: ô được chọn nếu vote_frac>=vote_thr, gom thành CÁC CỤM Ô
    LIỀN KỀ (4-neighbour). Mỗi cụm -> 1 bbox (1 khối u).
      multi=False -> trả bbox cụm tổng-vote lớn nhất (1 box, tương thích cũ).
      multi=True  -> trả LIST mọi bbox cụm (>=min_cluster_cells ô), TẤT CẢ khối u.
    vote_thr=None -> dùng global VOTE_THR (để sweep loop)."""
    if vote_thr is None:
        vote_thr = VOTE_THR
    cols, rows = rec["cols"], rec["rows"]
    sel = {ci for ci, v in vote_by_cell.items() if v >= vote_thr}
    if not sel and vote_by_cell:
        sel = {max(vote_by_cell, key=vote_by_cell.get)}
    if not sel:
        return [] if multi else None
    # gom TẤT CẢ cụm liền kề
    seen, clusters = set(), []
    for start in sel:
        if start in seen:
            continue
        stack, comp = [start], []
        while stack:
            ci = stack.pop()
            if ci in seen or ci not in sel:
                continue
            seen.add(ci); comp.append(ci)
            r, c = divmod(ci, cols)
            for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= nr < rows and 0 <= nc < cols:
                    stack.append(nr * cols + nc)
        clusters.append(comp)

    def box_of(comp):
        cells = [rec["cells"][ci] for ci in comp]
        return (min(c[0] for c in cells), min(c[1] for c in cells),
                max(c[2] for c in cells), max(c[3] for c in cells))

    clusters = [c for c in clusters if len(c) >= min_cluster_cells]
    clusters.sort(key=lambda comp: sum(vote_by_cell[ci] for ci in comp), reverse=True)
    if not clusters:
        return [] if multi else None
    if multi:
        return [box_of(c) for c in clusters]      # MỌI khối u
    return box_of(clusters[0])                     # cụm vote lớn nhất


def main():
    done = json.load(open(os.path.join(ROOT, "labels/done.json")))
    stems = [s for s in done if os.path.isfile(os.path.join(TUMOR_DIR, s + ".png"))]
    print(f"Ảnh có nhãn u: {len(stems)}")
    X, Y, groups, idx, per_img = build_dataset(stems)
    print(f"Ô hợp lệ: {len(X)} | dương (>= {POS_THR}): {int(Y.sum())} "
          f"({Y.mean()*100:.1f}%)")

    # CV theo bệnh nhân: 5 fold theo md5(patient)
    pats = sorted(set(groups))
    fold_of = {p: int(hashlib.md5(p.encode()).hexdigest(), 16) % 5 for p in pats}
    cell_auc = []
    sweep_votes = {}
    print(f"CV 5 fold, mỗi fold train {ENSEMBLE} model rồi biểu quyết đa số...")
    for k in range(5):
        tr = np.array([fold_of[g] != k for g in groups])
        te = ~tr
        if te.sum() == 0 or Y[tr].sum() == 0:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        models = train_ensemble(X[tr], Y[tr], mu, sd)
        vote_te, mean_te = ensemble_vote(models, X[te], mu, sd)
        cell_auc.append(auc(Y[te], mean_te))
        te_idx = [idx[i] for i in np.where(te)[0]]
        votes = {}
        for (s, ci), vv in zip(te_idx, vote_te):
            votes.setdefault(s, {})[ci] = float(vv)
        for s, vc in votes.items():
            sweep_votes.setdefault(s, {}).update(vc)
            per_img[s]["_present"] = True
    # quét ngưỡng biểu quyết VOTE_THR trên dự đoán out-of-fold
    print(f"\nAUC mức ô (out-of-fold): {np.nanmean(cell_auc):.3f}")
    print("\n=== Quét ngưỡng biểu quyết (CV theo bệnh nhân) ===")
    print(f"{'VOTE_THR':>8} | {'%u trong box(tv)':>16} | {'phủ>=90%':>8} | "
          f"{'chứa tâm':>8} | {'box/bp(tv)':>10} | {'IoU(tv)':>8}")
    global VOTE_THR
    best_cfg = None
    for vt in [0.5, 0.6, 0.7, 0.8, 0.9]:
        VOTE_THR = vt
        cov, contain, bfrac, ious = [], [], [], []
        for s, vc in sweep_votes.items():
            rec = per_img[s]; ab = auto_box(rec, vc); tb = rec["tumor_bbox"]
            ious.append(iou(ab, tb))
            if ab and tb:
                cxt = (tb[0] + tb[2]) / 2; cyt = (tb[1] + tb[3]) / 2
                contain.append(1.0 if (ab[0] <= cxt <= ab[2] and ab[1] <= cyt <= ab[3]) else 0.0)
                inside = sum(rec["tumor_px"][ci] for ci, c in enumerate(rec["cells"])
                             if c[0] >= ab[0] and c[2] <= ab[2] and c[1] >= ab[1] and c[3] <= ab[3])
                cov.append(inside / max(1, rec["tumor_total"]))
                bx0, by0, bx1, by1 = rec["spec_bbox"]
                bfrac.append((ab[2]-ab[0])*(ab[3]-ab[1]) / max(1, (bx1-bx0)*(by1-by0)))
        print(f"{vt:>8.1f} | {np.median(cov)*100:>15.1f}% | {(np.array(cov)>=0.9).mean()*100:>7.1f}% | "
              f"{np.mean(contain)*100:>7.1f}% | {np.median(bfrac)*100:>9.1f}% | {np.median(ious):>8.3f}")

    # train lại toàn bộ -> lưu ENSEMBLE để app dùng (chạy 100 lần + biểu quyết)
    save_vote_thr = float(os.environ.get("VOTE_THR", "0.60"))   # điểm vận hành mặc định
    mu, sd = X.mean(0), X.std(0) + 1e-6
    models = train_ensemble(X, Y, mu, sd)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    torch.save({"states": [m.state_dict() for m in models], "mu": mu, "sd": sd,
                "d": X.shape[1], "grid": GRID, "pos_thr": POS_THR,
                "prob_thr": PROB_THR, "vote_thr": save_vote_thr, "n_models": ENSEMBLE},
               os.path.join(ROOT, "models/cellbox_ensemble.pt"))
    print(f"\n-> models/cellbox_ensemble.pt ({ENSEMBLE} model, vote_thr={save_vote_thr})")


if __name__ == "__main__":
    main()
