"""
Sanity-check ĐÚNG CÁCH: diện tích KHỐI U (mask) quy về cm² rồi so với kích u Excel.

Spearman thô (mask_px ↔ Excel_cm²) vô nghĩa vì px/cm khác nhau từng ảnh.
Ở đây chuẩn hoá: tumor_cm2 = union_area_px / (px_per_cm**2), chỉ tính trên ảnh
CÓ px/cm (ruler dò được), rồi:
  - Spearman(tumor_cm2, mass_area_cm2 Excel)   -> kỳ vọng dương rõ
  - tỉ lệ tumor_cm2 / mass_area_cm2 (mask 2D nhỏ hơn diện tích từ caliper 3D
    bao nhiêu) để biết mask có hợp lý về độ lớn không.

Nguồn:
  labels/annotations.csv  (union_area_px = mask u; mass_area_cm2 = Excel)
  processed/metadata.csv  (px_per_cm mỗi ảnh)

Chạy: python area_sanity.py
"""
import csv
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def spearman(x, y):
    def rank(a):
        order = sorted(range(len(a)), key=lambda i: a[i])
        r = [0] * len(a)
        for ri, idx in enumerate(order):
            r[idx] = ri
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    if n < 3:
        return float("nan")
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def main():
    ann = {r["stem"]: r for r in csv.DictReader(open(os.path.join(ROOT, "labels/annotations.csv")))}
    meta = {r["stem"]: r for r in csv.DictReader(open(os.path.join(ROOT, "processed/metadata.csv")))}

    pairs = []          # (tumor_cm2, mass_cm2, ratio, stem)
    no_pxcm = no_mass = 0
    for stem, a in ann.items():
        m = meta.get(stem)
        pxcm = fnum(m["px_per_cm"]) if m else None
        area_px = fnum(a.get("union_area_px"))
        mass_cm2 = fnum(a.get("mass_area_cm2"))
        if not pxcm:
            no_pxcm += 1
            continue
        if not mass_cm2:
            no_mass += 1
            continue
        tumor_cm2 = area_px / (pxcm ** 2)
        pairs.append((tumor_cm2, mass_cm2, tumor_cm2 / mass_cm2, stem))

    print(f"Tổng nhãn: {len(ann)} | dùng được (có px/cm & Excel cm²): {len(pairs)}")
    print(f"  bỏ vì thiếu px/cm: {no_pxcm} | thiếu Excel cm²: {no_mass}")
    if len(pairs) >= 3:
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        ratios = sorted(p[2] for p in pairs)
        med = ratios[len(ratios) // 2]
        print(f"\nSpearman(tumor_cm2 ↔ Excel mass_cm2) = {rho:.3f}  (n={len(pairs)})")
        print(f"  >0.5 = mask to/nhỏ ĐÚNG theo kích u thật -> tin được")
        print(f"tỉ lệ tumor_cm2/Excel_cm2: med {med:.2f} "
              f"(min {ratios[0]:.2f}, max {ratios[-1]:.2f})")
        print(f"  ~0.5-1.0 hợp lý (mask 2D < diện tích từ caliper 3D).")
        big = [p for p in pairs if p[2] > 1.8]
        if big:
            print(f"\n[soi lại] {len(big)} ảnh mask LỚN bất thường (>1.8x Excel) — có thể còn dính mô lành:")
            for t, mm, ra, s in sorted(big, key=lambda p: -p[2])[:8]:
                print(f"  {s[:34]:34s} tumor={t:5.1f}cm2 Excel={mm:5.1f}cm2 x{ra:.1f}")
    else:
        print("Không đủ cặp để tính (px/cm dò được quá ít trên tập đã gán).")


if __name__ == "__main__":
    main()
