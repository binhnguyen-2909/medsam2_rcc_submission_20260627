"""
Gắn cờ nhãn cần SỬA LẠI: mask nghi "khoanh cả lát bệnh phẩm" thay vì riêng u.

Heuristic (không cần px/cm):
  - tách mô bệnh phẩm khỏi nền tím (HSV) -> specimen_area_px
  - fill_ratio = union_mask_area / specimen_area  (cao = mask phủ gần hết mô -> nghi cả lát)
  - cũng liệt kê last_score thấp (<0.85) đã accept.
In bảng ưu tiên + ghi labels/relabel_queue.csv.

  python flag_relabel.py
"""
import csv
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(ROOT, "data/20241212")
MASKS = os.path.join(ROOT, "labels/masks")
CSV = os.path.join(ROOT, "labels/annotations.csv")
OUT = os.path.join(ROOT, "labels/relabel_queue.csv")


def specimen_area(bgr):
    """Diện tích MÔ bệnh phẩm: loại nền tím + vùng trắng (thước/nhãn) + dải đáy."""
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    purple = (h > 110) & (h < 145) & (s > 90)      # nền tím
    whitish = (s < 55) & (v > 170)                  # thước + nhãn ID (trắng/xám)
    tissue = ~(purple | whitish)
    tissue[int(H * 0.86):, :] = False               # bỏ dải đáy (thước)
    tissue = cv2.morphologyEx(tissue.astype(np.uint8), cv2.MORPH_OPEN,
                              np.ones((9, 9), np.uint8))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return int(tissue.sum())


def main():
    rows = list(csv.DictReader(open(CSV)))
    out = []
    for r in rows:
        stem = r["stem"]
        mp = os.path.join(MASKS, stem + ".png")
        ip = os.path.join(IMG, stem + ".jpg")
        if not (os.path.isfile(mp) and os.path.isfile(ip)):
            continue
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127
        bgr = cv2.imread(ip)
        spec = specimen_area(bgr)
        ua = int(m.sum())
        fr = ua / spec if spec else 0.0
        sc = float(r["last_score"]) if r.get("last_score") not in (None, "") else None
        flags = []
        if fr > 0.75:
            flags.append("nghi-cả-lát")
        if sc is not None and sc < 0.85:
            flags.append("score-thấp")
        out.append({"stem": stem, "patient_id": r["patient_id"],
                    "fill_ratio": round(fr, 3), "union_area_px": ua,
                    "specimen_px": spec, "last_score": r.get("last_score", ""),
                    "n_objects": r.get("n_objects", ""),
                    "mass_dims_cm": r.get("mass_dims_cm", ""),
                    "flags": "+".join(flags)})

    out.sort(key=lambda d: (-len(d["flags"]), -d["fill_ratio"]))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    flagged = [d for d in out if d["flags"]]
    print(f"== {len(out)} nhãn | {len(flagged)} cần xem lại ==\n")
    print(f"{'stem':36s} {'fill':>5s} {'score':>6s} {'u_cm':>9s}  cờ")
    for d in out:
        mark = "  <<" if d["flags"] else ""
        print(f"{d['stem'][:36]:36s} {d['fill_ratio']:5.2f} "
              f"{str(d['last_score']):>6s} {str(d['mass_dims_cm']):>9s}  "
              f"{d['flags']}{mark}")
    print(f"\n-> {OUT}")
    print("fill_ratio = mask/diện-tích-mô. >0.75 = nghi khoanh cả lát (sửa bó sát u).")


if __name__ == "__main__":
    main()
