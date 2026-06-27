"""
Lọc ảnh MẶT-CẮT (cut-surface, có thể thấy khối u) vs MẶT-NGOÀI (bao thận trơn)
trên toàn bộ 1392 ảnh, dùng cut_surface_score trong processed/metadata.csv.

Ngưỡng KHÔNG đặt tùy tiện: hiệu chỉnh bằng 55 ảnh ĐÃ GÁN (labels/done.json) —
tất cả đều là mặt cắt (vì có khối u để khoanh). Lấy phân vị thấp của nhóm này
làm ngưỡng giữ (recall cao trên nhóm chắc chắn là mặt cắt).

Xuất:
  processed/cut_surface_filter.csv   stem, cut_surface_score, px_per_cm, is_cut_surface, labeled
  results/cut_filter_montage.jpg     ảnh quanh ngưỡng (để QC mắt thường ngưỡng)
  In phân bố + ngưỡng + số ảnh giữ.

Chạy: python filter_cut_surface.py [--q 0.05]   (q = phân vị thấp của nhóm đã gán)
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(ROOT, "processed/metadata.csv")
IMG_DIR = os.path.join(ROOT, "data/20241212")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=0.05,
                    help="phân vị thấp của cut_score nhóm đã gán -> ngưỡng giữ")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(META)))
    for r in rows:
        r["cut"] = float(r["cut_surface_score"]) if r["cut_surface_score"] else 0.0

    done = set(json.load(open(os.path.join(ROOT, "labels/done.json"))))
    labeled = [r for r in rows if r["stem"] in done]
    lab_scores = sorted(r["cut"] for r in labeled)
    if not lab_scores:
        raise SystemExit("Không có ảnh đã gán để hiệu chỉnh ngưỡng.")

    thr = float(np.quantile(lab_scores, args.q))
    for r in rows:
        r["is_cut"] = r["cut"] >= thr

    n_keep = sum(r["is_cut"] for r in rows)
    # độ phủ: bao nhiêu ảnh đã-gán bị ngưỡng giữ lại (kỳ vọng ~ 1-q)
    lab_keep = sum(1 for r in labeled if r["cut"] >= thr)

    with open(os.path.join(ROOT, "processed/cut_surface_filter.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "cut_surface_score", "px_per_cm", "is_cut_surface", "labeled"])
        for r in rows:
            w.writerow([r["stem"], r["cut_surface_score"], r["px_per_cm"],
                        int(r["is_cut"]), int(r["stem"] in done)])

    # montage ảnh quanh ngưỡng (12 ảnh ngay dưới & 12 ngay trên) để QC ngưỡng
    near = sorted(rows, key=lambda r: abs(r["cut"] - thr))[:24]
    near = sorted(near, key=lambda r: r["cut"])
    tiles = []
    for r in near:
        p = os.path.join(IMG_DIR, r["stem"] + ".jpg")
        im = cv2.imread(p)
        if im is None:
            continue
        im = cv2.resize(im, (320, 213))
        col = (0, 200, 0) if r["is_cut"] else (0, 0, 230)
        cv2.rectangle(im, (0, 0), (319, 212), col, 6)
        cv2.putText(im, f"{r['cut']:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 3, cv2.LINE_AA)
        tiles.append(im)
    if tiles:
        cols = 6
        rows_n = (len(tiles) + cols - 1) // cols
        while len(tiles) < cols * rows_n:
            tiles.append(np.zeros((213, 320, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[i*cols:(i+1)*cols]) for i in range(rows_n)])
        os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
        cv2.imwrite(os.path.join(ROOT, "results/cut_filter_montage.jpg"), grid)

    print(f"Ngưỡng cut_surface_score = {thr:.3f}  (phân vị {args.q:.0%} của 55 ảnh đã gán)")
    print(f"Nhóm đã gán: min {lab_scores[0]:.2f} / med {np.median(lab_scores):.2f} "
          f"/ max {lab_scores[-1]:.2f}  -> giữ {lab_keep}/{len(labeled)}")
    print(f"Toàn bộ: giữ {n_keep}/{len(rows)} ảnh là MẶT-CẮT "
          f"({100*n_keep/len(rows):.0f}%), loại {len(rows)-n_keep} (mặt-ngoài/nghi)")
    print("-> processed/cut_surface_filter.csv ; results/cut_filter_montage.jpg")


if __name__ == "__main__":
    main()
