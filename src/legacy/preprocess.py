"""
Tiền xử lý ảnh đại thể RCC (gross pathology) cho pipeline MedSAM2.

Mỗi ảnh:
  1. Tách nền tím (HSV)            -> mask bệnh phẩm (specimen)
  2. Lấy connected-component lớn nhất -> loại nhãn ID + thước + nhiễu
  3. Dò thước ở đáy ảnh           -> pixels/cm (FFT trên profile vạch thước)
  4. Tính điểm "mặt cắt"          -> cờ gợi ý ảnh cut-surface (cần QC tay)
  5. Xuất: crop sạch (nền đen) + mask + metadata.csv + ảnh QC overlay

Lưu ý quan trọng (xem phản biện):
  - pixels/cm lấy từ THƯỚC trong ảnh, KHÔNG phải từ Excel.
  - --tick_cm là cm cho MỖI khoảng vạch phát hiện được; PHẢI hiệu chỉnh tay
    bằng cách xem ảnh QC rồi chỉnh lại cho khớp thước thật.
  - Bước lọc mặt-cắt chỉ là gợi ý, không phải chân lý.

Ví dụ:
  python preprocess.py --img_dir data/20241212 --out_dir processed \
      --sample 12 --debug --tick_cm 1.0
"""
import argparse
import csv
import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np

# ── Tham số nền tím (đo thực tế trên dataset: H~126-132, S~200, V~255) ──────
BG_HUE_LO, BG_HUE_HI = 112, 142
BG_SAT_MIN = 80
BG_VAL_MIN = 80


def parse_patient_id(stem: str) -> str:
    """'SS21-33569^2021_06_01_..' -> 'SS21-33569'"""
    return stem.split("^")[0].strip()


def specimen_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Trả về mask nhị phân (uint8 0/1) của bệnh phẩm = thành phần foreground lớn nhất."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H_, S_, V_ = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bg = ((H_ >= BG_HUE_LO) & (H_ <= BG_HUE_HI) & (S_ >= BG_SAT_MIN) & (V_ >= BG_VAL_MIN))
    fg = (~bg).astype(np.uint8)

    # Dọn nhiễu: đóng lỗ rồi mở để bỏ đốm nhỏ
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return np.zeros(fg.shape, np.uint8)
    # Bỏ nền (label 0); chọn component diện tích lớn nhất = bệnh phẩm
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    mask = (labels == best).astype(np.uint8)
    # Lấp lỗ bên trong bệnh phẩm (mạch máu, bóng)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=2)
    return mask


def detect_ruler_pxcm(img_bgr: np.ndarray, specimen: np.ndarray,
                      min_lag: int = 20, max_lag: int = 400):
    """
    Dò thước ở đáy ảnh bằng AUTOCORRELATION (ổn định hơn FFT vì không bị
    nhiễu sóng hài mm/cm). Trả về (period_px, confidence, debug_dict).
    period_px = khoảng cách pixel giữa các vạch (nhân --tick_cm để ra px/cm).
    confidence in [0,1] = độ cao đỉnh autocorrelation đầu tiên.
    """
    H, W = img_bgr.shape[:2]
    y0 = int(0.85 * H)
    band = img_bgr[y0:H]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Hàng thuộc thước = hàng nhiều pixel sáng (thước trắng), không bị bệnh phẩm che
    spec_band = specimen[y0:H]
    bright = ((gray > 140) & (spec_band == 0)).astype(np.float32)
    row_bright = bright.mean(axis=1)
    ruler_rows = np.where(row_bright > 0.35)[0]
    dbg = {"band_y0": y0}
    if len(ruler_rows) < 5:
        return None, 0.0, dbg
    r0, r1 = int(ruler_rows.min()), int(ruler_rows.max())
    dbg["ruler_y0"], dbg["ruler_y1"] = y0 + r0, y0 + r1

    # Profile "độ tối" theo cột trong dải thước -> vạch thước tạo dao động tuần hoàn
    sub = gray[r0:r1 + 1]
    col_valid = (sub > 120).mean(axis=0) > 0.4   # chỉ cột nằm trên thước trắng
    if col_valid.sum() < W * 0.2:
        return None, 0.0, dbg
    cols = np.where(col_valid)[0]
    c0, c1 = cols.min(), cols.max()
    dark = (255.0 - sub).copy()
    dark[sub > 200] = 0.0                         # nền trắng -> 0, chỉ giữ vạch đen
    prof = dark[:, c0:c1 + 1].mean(axis=0)
    prof = prof - prof.mean()
    if prof.std() < 1e-3:
        return None, 0.0, dbg

    # Autocorrelation chuẩn hoá
    ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
    ac = ac / (ac[0] + 1e-9)
    hi = min(max_lag, len(ac) - 2)
    # đỉnh cục bộ đầu tiên đủ mạnh sau min_lag = chu kỳ cơ bản (khoảng cách vạch)
    best_lag, best_val = None, 0.0
    for l in range(min_lag, hi):
        if ac[l] > ac[l - 1] and ac[l] > ac[l + 1] and ac[l] > 0.3 and ac[l] > best_val:
            best_lag, best_val = l, float(ac[l])
            break
    if best_lag is None:
        return None, 0.0, dbg
    dbg["period_px"] = float(best_lag)
    return float(best_lag), round(best_val, 3), dbg


def cut_surface_score(img_bgr: np.ndarray, mask: np.ndarray) -> float:
    """
    Điểm gợi ý mặt-cắt (cao = nhiều khả năng là ảnh cut-surface).
    Mặt cắt có mô vàng/nâu nhạt không đồng nhất; mặt ngoài (bao thận) đỏ-nâu trơn.
    Dùng độ lệch chuẩn Hue/Value bên trong bệnh phẩm (độ không đồng nhất).
    """
    if mask.sum() < 100:
        return 0.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = mask.astype(bool)
    hue_std = float(np.std(hsv[..., 0][m]))
    val_std = float(np.std(hsv[..., 2][m]))
    # vùng "tan/yellow" mô cắt: Hue ~ 10-35
    h = hsv[..., 0][m]
    tan_frac = float(((h >= 10) & (h <= 35)).mean())
    return round(0.5 * (val_std / 60.0) + 0.3 * (hue_std / 25.0) + 0.2 * tan_frac, 3)


def make_qc(img_bgr, mask, bbox, period, pxcm, slice_cm2, rdbg):
    out = img_bgr.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 230, 0), 4)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 255), 3)
    if "ruler_y0" in rdbg:
        cv2.rectangle(out, (0, rdbg["ruler_y0"]), (out.shape[1] - 1, rdbg["ruler_y1"]),
                      (255, 80, 80), 3)
    lines = []
    if period:
        lines.append(f"tick_period={period:.1f}px")
    if pxcm:
        lines.append(f"px/cm={pxcm:.1f}")
    if slice_cm2:
        lines.append(f"slice~{slice_cm2:.1f}cm2")
    for i, t in enumerate(lines):
        cv2.putText(out, t, (20, 60 + i * 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(out, t, (20, 60 + i * 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (255, 255, 255), 3, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default="data/20241212")
    ap.add_argument("--out_dir", default="processed")
    ap.add_argument("--tick_cm", type=float, default=1.0,
                    help="cm cho mỗi khoảng vạch thước phát hiện được (HIỆU CHỈNH TAY)")
    ap.add_argument("--sample", type=int, default=None, help="chỉ xử lý N ảnh đầu")
    ap.add_argument("--debug", action="store_true", help="xuất ảnh QC overlay")
    ap.add_argument("--bg_black", action="store_true", default=True,
                    help="bôi đen nền trong ảnh crop")
    args = ap.parse_args()

    crop_dir = os.path.join(args.out_dir, "crop")
    mask_dir = os.path.join(args.out_dir, "mask")
    qc_dir = os.path.join(args.out_dir, "qc")
    for d in (crop_dir, mask_dir, qc_dir):
        os.makedirs(d, exist_ok=True)

    imgs = sorted(glob(os.path.join(args.img_dir, "*.jpg")))
    if args.sample:
        imgs = imgs[: args.sample]
    print(f"Xử lý {len(imgs)} ảnh ...")

    rows = []
    for p in imgs:
        stem = Path(p).stem
        img = cv2.imread(p)
        if img is None:
            print(f"[skip] đọc lỗi {stem}")
            continue
        H, W = img.shape[:2]
        mask = specimen_mask(img)
        area_px = int(mask.sum())
        if area_px < 1000:
            print(f"[warn] không thấy bệnh phẩm: {stem}")
            bbox = None
        else:
            ys, xs = np.where(mask)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        period, rconf, rdbg = detect_ruler_pxcm(img, mask)
        pxcm = (period / args.tick_cm) if period else None
        # Diện tích CẢ lát cắt bệnh phẩm (cả quả thận) — CHỈ để kiểm tra px/cm
        # có hợp lý không (~20-60cm²). KHÔNG so với số cm trong Excel: Excel là
        # kích thước KHỐI U, không phải lát cắt.
        slice_cm2 = (area_px / (pxcm ** 2)) if pxcm else None

        cut = cut_surface_score(img, mask)

        # Xuất crop sạch
        if bbox is not None:
            out = img.copy()
            if args.bg_black:
                out[mask == 0] = 0
            x0, y0, x1, y1 = bbox
            cv2.imwrite(os.path.join(crop_dir, stem + ".jpg"), out[y0:y1 + 1, x0:x1 + 1])
            cv2.imwrite(os.path.join(mask_dir, stem + ".png"), mask * 255)

        if args.debug:
            qc = make_qc(img, mask, bbox, period, pxcm, slice_cm2, rdbg)
            cv2.imwrite(os.path.join(qc_dir, stem + ".jpg"),
                        cv2.resize(qc, (W // 2, H // 2)))

        rows.append({
            "stem": stem, "patient_id": parse_patient_id(stem),
            "W": W, "H": H, "area_px": area_px,
            "bbox": "|".join(map(str, bbox)) if bbox else "",
            "tick_period_px": round(period, 2) if period else "",
            "ruler_conf": round(rconf, 3),
            "px_per_cm": round(pxcm, 2) if pxcm else "",
            "slice_area_cm2": round(slice_cm2, 2) if slice_cm2 else "",  # CẢ lát cắt, KHÔNG phải kích u Excel
            "cut_surface_score": cut,
        })
        print(f"[ok] {stem[:22]:22s} area={area_px:>8d}px "
              f"px/cm={pxcm if pxcm else 'NA':>6} conf={rconf:.2f} cut={cut:.2f}")

    csv_path = os.path.join(args.out_dir, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nMetadata -> {csv_path}")
    if args.debug:
        print(f"QC overlay -> {qc_dir}/  (xem để hiệu chỉnh --tick_cm)")


if __name__ == "__main__":
    main()
