"""
Đánh giá tập nhãn đã gán (labels/annotations.csv) để quyết định:
SAM2.1@1024 box->mask zero-shot đã đủ tốt chưa, hay cần fine-tune?

Không có ground-truth tuyệt đối (mask = do người accept), nên đánh giá:
  1. Thống kê sam_score; gắn cờ ca score thấp (<0.85) cần soi lại.
  2. Bao nhiêu ca phải tinh chỉnh điểm (n_pos/n_neg>0) -> box-only fail rate.
  3. Sanity-check: tương quan hạng (Spearman) mask_area_px ↔ mass_area_cm2 Excel
     (kỳ vọng dương; px/cm khác nhau từng ảnh nên chỉ là tín hiệu yếu).
  4. Montage overlay -> results/labels_montage.jpg để QC mắt thường.

  python eval_labels.py
"""
import csv
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(ROOT, "labels/annotations.csv")
OV = os.path.join(ROOT, "labels/overlays")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)


def spearman(x, y):
    def rank(a):
        order = sorted(range(len(a)), key=lambda i: a[i])
        r = [0] * len(a)
        for rank_i, idx in enumerate(order):
            r[idx] = rank_i
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    if n < 3:
        return float("nan")
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def main():
    rows = list(csv.DictReader(open(CSV)))
    n = len(rows)
    print(f"== Tập nhãn: {n} ảnh, {len(set(r['patient_id'] for r in rows))} bệnh nhân ==\n")

    def fscore(r):
        v = r.get("last_score", "")
        return float(v) if v not in ("", None) else None

    scores = [s for s in (fscore(r) for r in rows) if s is not None]
    print("-- SAM score (last_score; bỏ ô trống) --")
    if scores:
        ns = len(scores)
        print(f"  có score: {ns}/{n} | min {min(scores):.3f} | "
              f"median {sorted(scores)[ns//2]:.3f} | mean {sum(scores)/ns:.3f} | "
              f"max {max(scores):.3f}")
        low = [r for r in rows if fscore(r) is not None and fscore(r) < 0.85]
        print(f"  ca score <0.85 ({len(low)}/{ns}) — soi lại overlay:")
        for r in sorted(low, key=fscore):
            print(f"    {r['last_score']}  {r['stem'][:34]}  area={r['union_area_px']}px")

    # sanity-check diện tích
    pairs = [(float(r["union_area_px"]), float(r["mass_area_cm2"]))
             for r in rows if r["mass_area_cm2"]]
    print("\n-- Sanity-check diện tích (Spearman mask_px ↔ mass_cm² Excel) --")
    if len(pairs) >= 3:
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        print(f"  rho = {rho:.3f}  (n={len(pairs)}; >0.5 = mask to/nhỏ đi đúng "
              f"theo kích u, hợp lý; px/cm khác ảnh nên không kỳ vọng ~1)")
    else:
        print(f"  thiếu dữ liệu Excel ({len(pairs)} ca)")

    # montage overlay
    files = [os.path.join(OV, r["stem"] + ".jpg") for r in rows]
    files = [f for f in files if os.path.isfile(f)]
    if files:
        cell = 360
        cols = 5
        rows_n = (len(files) + cols - 1) // cols
        canvas = np.full((rows_n * cell, cols * cell, 3), 30, np.uint8)
        for i, f in enumerate(files):
            im = cv2.imread(f)
            h, w = im.shape[:2]
            s = cell / max(h, w)
            im = cv2.resize(im, (int(w * s), int(h * s)))
            rr, cc = divmod(i, cols)
            canvas[rr*cell:rr*cell+im.shape[0], cc*cell:cc*cell+im.shape[1]] = im
        p = os.path.join(OUT, "labels_montage.jpg")
        cv2.imwrite(p, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"\nMontage QC -> {p} ({len(files)} ảnh)")


if __name__ == "__main__":
    main()
