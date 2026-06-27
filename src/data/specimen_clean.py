"""
Tách BỆNH PHẨM đáng tin, CẮT THƯỚC + NHÃN trước.

Nền = TÍM (hue rất ổn định ~127, bão hoà cao) -> tách theo hue (không dùng
'CC lớn nhất toàn ảnh' vì thước trắng dài hay bị chọn nhầm khi mô nhỏ).
Phần not-purple = mô + thước + nhãn. Loại 2 thứ rác theo vị trí/hình dạng:
  - THƯỚC : component RỘNG NGANG nằm ở ĐÁY ảnh.
  - NHÃN  : component nằm gọn ở GÓC TRÊN-TRÁI.
Còn lại -> CC lớn nhất = bệnh phẩm.

clean_specimen(bgr) -> (mask uint8 HxW, bbox (x0,y0,x1,y1), dbg)
"""
import cv2
import numpy as np


def _bg_hue(hsv):
    """Hue nền: lấy mẫu viền ngoài, BỎ góc trên-trái (nhãn) và dải đáy (thước)."""
    h, w = hsv.shape[:2]
    border = np.zeros((h, w), bool)
    b = max(6, int(0.06 * min(h, w)))
    border[:b, :] = border[-b:, :] = border[:, :b] = border[:, -b:] = True
    border[:int(0.25 * h), :int(0.35 * w)] = False     # bỏ vùng nhãn
    border[int(0.85 * h):, :] = False                  # bỏ dải thước
    hs = hsv[:, :, 0][border]
    return float(np.median(hs)) if len(hs) else 127.0


def _purple_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h0 = _bg_hue(hsv)
    H = hsv[:, :, 0].astype(int)
    dh = np.minimum(np.abs(H - h0), 180 - np.abs(H - h0))   # vòng hue
    purple = (dh <= 22) & (hsv[:, :, 1] > 60)
    return purple.astype(np.uint8), h0


def clean_specimen(bgr):
    h, w = bgr.shape[:2]
    purple, h0 = _purple_mask(bgr)
    fg = (purple == 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    n, lab, stats, cent = cv2.connectedComponentsWithStats(fg, connectivity=8)
    dropped = {"ruler": [], "label": []}
    keep = np.zeros((h, w), np.uint8)
    cand = []   # (area, comp_id)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < 0.0008 * h * w:
            continue
        cx, cy = cent[i]
        is_ruler = (ww > 0.45 * w) and (y > 0.70 * h)          # rộng ngang & ở đáy
        is_label = (cx < 0.35 * w) and (cy < 0.22 * h) and \
                   (ww < 0.40 * w) and (hh < 0.22 * h)         # gọn ở góc TL
        if is_ruler:
            dropped["ruler"].append((int(x), int(y), int(ww), int(hh))); continue
        if is_label:
            dropped["label"].append((int(x), int(y), int(ww), int(hh))); continue
        cand.append((area, i))
    if not cand:
        return np.zeros((h, w), np.uint8), (0, 0, w - 1, h - 1), {"h0": h0, **dropped}
    # GIỮ UNION TẤT CẢ mảnh mô (bệnh phẩm hay nhiều mảnh) - không chỉ mảnh lớn nhất
    keep_ids = np.array([i for _, i in cand])
    keep = np.isin(lab, keep_ids).astype(np.uint8)
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    ys, xs = np.where(keep)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return keep, bbox, {"h0": h0, **dropped}
