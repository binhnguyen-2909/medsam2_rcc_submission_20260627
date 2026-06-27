"""
DELIVERABLE — box→mask cho ảnh đại thể RCC bằng SAM2.1 vanilla @1024 (ZERO-SHOT).

Người dùng vẽ 1 (hoặc nhiều) bounding-box quanh KHỐI U → script sinh mask nhị phân.
Nhiều box/ảnh (ảnh nhiều mảnh u) → hợp (union) thành 1 mask.

KHÔNG fine-tune (đã chứng minh FT không vượt zero-shot tới 1008 ảnh train).

------------------------------------------------------------------ CÁCH DÙNG
1 ảnh, 1 box:
  python predict_box2mask.py --image a.jpg --box "x0,y0,x1,y1" --out a_mask.png

1 ảnh, nhiều box (lặp --box):
  python predict_box2mask.py --image a.jpg \
      --box "120,80,400,350" --box "600,500,820,700" \
      --out a_mask.png --overlay a_overlay.jpg

Batch từ CSV (cột: image,x0,y0,x1,y1 ; NHIỀU dòng cùng image -> union):
  python predict_box2mask.py --csv boxes.csv --out_dir out_masks [--overlay_dir out_ov]

Toạ độ box theo PIXEL trên ảnh GỐC (x0,y0 = góc trên-trái; x1,y1 = góc dưới-phải).
Output: PNG nhị phân full-res (0=nền, 255=u), cùng kích thước ảnh gốc.

Cấu hình model (env, mặc định = bản đã chốt):
  SAM2_CONFIG=configs/sam2.1_hiera_t512  SAM2_CKPT=checkpoints/sam2.1_hiera_tiny.pt  SAM2_RES=1024
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CONFIG = os.environ.get("SAM2_CONFIG", "configs/sam2.1_hiera_t512")
CKPT = os.environ.get("SAM2_CKPT", "checkpoints/sam2.1_hiera_tiny.pt")
RES = int(os.environ.get("SAM2_RES", "1024"))


def load_predictor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(CONFIG, CKPT, device=device,
                       hydra_overrides_extra=[f"++model.image_size={RES}"])
    pred = SAM2ImagePredictor(model)
    ac = (torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda"
          else torch.autocast("cpu", enabled=False))
    return pred, ac


def boxes_to_mask(pred, ac, rgb, boxes):
    """rgb: HxWx3 uint8. boxes: list [x0,y0,x1,y1]. -> mask bool HxW (union)."""
    H, W = rgb.shape[:2]
    union = np.zeros((H, W), bool)
    with torch.inference_mode(), ac:
        pred.set_image(rgb)
        for box in boxes:
            bx = np.array(box, dtype=np.float32)
            masks, scores, _ = pred.predict(box=bx, multimask_output=True)
            union |= masks[int(np.argmax(scores))].astype(bool)
    return union


def clip_box(box, W, H):
    x0, y0, x1, y1 = box
    x0, x1 = sorted((max(0, min(W - 1, int(x0))), max(0, min(W, int(x1)))))
    y0, y1 = sorted((max(0, min(H - 1, int(y0))), max(0, min(H, int(y1)))))
    return [x0, y0, x1, y1]


def make_overlay(rgb, mask, boxes):
    out = rgb.copy()
    out[mask] = (0.5 * out[mask] + 0.5 * np.array([0, 255, 120])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 0), 4)
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(out, (x0, y0), (x1, y1), (60, 120, 255), 4)
    return out


def parse_box(s):
    parts = [float(v) for v in s.replace(";", ",").replace(" ", ",").split(",") if v != ""]
    if len(parts) != 4:
        raise ValueError(f"box phải có 4 số x0,y0,x1,y1 — nhận: {s!r}")
    return parts


def run_one(pred, ac, img_path, boxes, out_path, overlay_path=None):
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"[bỏ] không đọc được ảnh: {img_path}")
        return False
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    boxes = [clip_box(b, W, H) for b in boxes]
    mask = boxes_to_mask(pred, ac, rgb, boxes)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, (mask.astype(np.uint8) * 255))
    if overlay_path:
        os.makedirs(os.path.dirname(os.path.abspath(overlay_path)), exist_ok=True)
        ov = make_overlay(rgb, mask, boxes)
        cv2.imwrite(overlay_path, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"[ok] {os.path.basename(img_path)} | {len(boxes)} box | "
          f"u={int(mask.sum())}px ({100*mask.sum()/(H*W):.1f}% ảnh) -> {out_path}")
    return True


def read_csv_boxes(path):
    """CSV cột: image,x0,y0,x1,y1 (nhiều dòng cùng image -> gom box). image có thể
    là đường dẫn tuyệt đối hoặc tương đối so với thư mục script / data/20241212."""
    by_img = {}
    for r in csv.DictReader(open(path)):
        img = r["image"].strip()
        box = [float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])]
        by_img.setdefault(img, []).append(box)
    return by_img


def resolve_image(img):
    for cand in (img, os.path.join(ROOT, img),
                 os.path.join(ROOT, "data/20241212", img),
                 os.path.join(ROOT, "data/20241212", img + ".jpg")):
        if os.path.isfile(cand):
            return cand
    return img


def main():
    ap = argparse.ArgumentParser(description="SAM2.1 zero-shot box->mask (RCC gross)")
    ap.add_argument("--image", help="1 ảnh")
    ap.add_argument("--box", action="append", default=[],
                    help='box "x0,y0,x1,y1" (lặp nhiều lần cho nhiều u)')
    ap.add_argument("--out", help="đường dẫn mask PNG (chế độ 1 ảnh)")
    ap.add_argument("--overlay", help="đường dẫn overlay QC (tuỳ chọn)")
    ap.add_argument("--csv", help="batch: CSV cột image,x0,y0,x1,y1")
    ap.add_argument("--out_dir", help="thư mục mask cho batch")
    ap.add_argument("--overlay_dir", help="thư mục overlay cho batch (tuỳ chọn)")
    args = ap.parse_args()

    pred, ac = load_predictor()

    if args.csv:
        if not args.out_dir:
            ap.error("--csv cần kèm --out_dir")
        by_img = read_csv_boxes(args.csv)
        print(f"Batch: {len(by_img)} ảnh từ {args.csv}")
        n = 0
        for img, boxes in by_img.items():
            ip = resolve_image(img)
            stem = os.path.splitext(os.path.basename(ip))[0]
            outp = os.path.join(args.out_dir, stem + ".png")
            ovp = (os.path.join(args.overlay_dir, stem + ".jpg")
                   if args.overlay_dir else None)
            n += run_one(pred, ac, ip, boxes, outp, ovp)
        print(f"Xong: {n}/{len(by_img)} ảnh có mask -> {args.out_dir}")
    else:
        if not (args.image and args.box and args.out):
            ap.error("chế độ 1 ảnh cần --image, ít nhất 1 --box, và --out")
        boxes = [parse_box(b) for b in args.box]
        run_one(pred, ac, resolve_image(args.image), boxes, args.out, args.overlay)


if __name__ == "__main__":
    main()
