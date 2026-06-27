"""
Sinh ỨNG VIÊN mask tự động (AMG tối giản) bằng SAM2.1@1024 để TĂNG TỐC gán nhãn:
gieo lưới điểm trong bệnh phẩm -> SAM2 sinh mask cho từng điểm -> lọc kích thước/
nằm-trong-specimen -> NMS bỏ trùng -> giữ top-K. Người chỉ việc BẤM chọn ứng viên
trúng khối u (thay vì vẽ box).

  python propose_amg.py --demo 4         # thử 4 ảnh mặt-cắt -> results/amg_demo.jpg
  python propose_amg.py --run            # batch toàn ảnh mặt-cắt chưa gán -> labels/proposals/

Lưu mỗi ảnh: labels/proposals/<stem>.npz  (masks uint8 KxHxW, scores, points)
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from preprocess import specimen_mask

IMG_DIR = os.path.join(ROOT, "data/20241212")
CONFIG = "configs/sam2.1_hiera_t512"
CKPT = "checkpoints/sam2.1_hiera_tiny.pt"
RES = 1024
PROP_DIR = os.path.join(ROOT, "labels/proposals")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_AC = (torch.autocast("cuda", dtype=torch.bfloat16) if DEVICE == "cuda"
       else torch.autocast("cpu", enabled=False))


def build_predictor():
    print(f"Nạp SAM2.1@{RES} trên {DEVICE} ...")
    m = build_sam2(CONFIG, CKPT, device=DEVICE,
                   hydra_overrides_extra=[f"++model.image_size={RES}"])
    return SAM2ImagePredictor(m)


def propose(P, rgb, spec, n_side=24, topk=8, batch=64,
            min_frac=0.01, max_frac=0.45, inside_min=0.85, nms_iou=0.6):
    H, W = rgb.shape[:2]
    with torch.inference_mode(), _AC:
        P.set_image(rgb)
    specb = spec.astype(bool)
    specarea = float(specb.sum())
    if specarea < 1000:
        return []
    fy = np.linspace(0.04, 0.96, n_side)
    fx = np.linspace(0.04, 0.96, n_side)
    pts = []
    for y in fy:
        for x in fx:
            px, py = int(x * W), int(y * H)
            if specb[py, px]:
                pts.append([px, py])
    if not pts:
        return []
    pts = np.array(pts, dtype=np.float32)

    cand = []
    for i in range(0, len(pts), batch):
        chunk = pts[i:i + batch]
        coords = torch.as_tensor(chunk.reshape(-1, 1, 2), device=DEVICE, dtype=torch.float)
        tc = P._transforms.transform_coords(coords, normalize=True, orig_hw=P._orig_hw[-1])
        labels = torch.ones((tc.shape[0], 1), dtype=torch.int, device=DEVICE)
        with torch.inference_mode(), _AC:
            masks, iou, _ = P._predict(tc, labels, multimask_output=True)
        masks = masks.cpu().numpy()                 # (b,3,H,W) bool
        iou = iou.float().cpu().numpy()             # (b,3)
        for b in range(masks.shape[0]):
            j = int(np.argmax(iou[b]))
            m = masks[b, j]
            a = float(m.sum())
            if a < min_frac * specarea or a > max_frac * specarea:
                continue
            if (m & specb).sum() / max(a, 1) < inside_min:
                continue
            cand.append((float(iou[b, j]), m, chunk[b]))

    cand.sort(key=lambda c: -c[0])
    keep = []
    for sc, m, pt in cand:
        if all(((m & k[1]).sum() / max((m | k[1]).sum(), 1)) <= nms_iou for k in keep):
            keep.append((sc, m, pt))
        if len(keep) >= topk:
            break
    return keep


def cut_unlabeled_stems():
    done = set(json.load(open(os.path.join(ROOT, "labels/done.json"))))
    skip = set(json.load(open(os.path.join(ROOT, "labels/skipped.json"))))
    out = []
    fp = os.path.join(ROOT, "processed/cut_surface_filter.csv")
    for r in csv.DictReader(open(fp)):
        if r["is_cut_surface"] == "1" and r["stem"] not in done and r["stem"] not in skip:
            out.append(r["stem"])
    return out


PALETTE = [(0, 255, 0), (255, 80, 80), (80, 160, 255), (255, 255, 0),
           (255, 0, 255), (0, 255, 255), (255, 150, 0), (150, 0, 255)]


def draw_candidates(rgb, keep):
    out = rgb.copy()
    for idx, (sc, m, pt) in enumerate(keep):
        col = PALETTE[idx % len(PALETTE)]
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, col, 5)
        cv2.putText(out, str(idx + 1), (int(pt[0]), int(pt[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 5, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    P = build_predictor()
    stems = cut_unlabeled_stems()
    print(f"Ảnh mặt-cắt chưa gán: {len(stems)}")

    if args.demo:
        sel = stems[:args.demo]
        tiles = []
        for s in sel:
            img = cv2.imread(os.path.join(IMG_DIR, s + ".jpg"))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            spec = specimen_mask(img)
            keep = propose(P, rgb, spec, topk=args.topk)
            vis = cv2.cvtColor(draw_candidates(rgb, keep), cv2.COLOR_RGB2BGR)
            vis = cv2.resize(vis, (640, 426))
            cv2.putText(vis, f"{s[:24]} | {len(keep)} ứng viên", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(vis)
            print(f"  {s[:30]:30s} -> {len(keep)} ứng viên "
                  f"(score {', '.join(f'{k[0]:.2f}' for k in keep[:5])})")
        cols = 2
        while len(tiles) % cols:
            tiles.append(np.zeros((426, 640, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[i:i+cols]) for i in range(0, len(tiles), cols)])
        os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
        cv2.imwrite(os.path.join(ROOT, "results/amg_demo.jpg"), grid)
        print("-> results/amg_demo.jpg")
        return

    if args.run:
        os.makedirs(PROP_DIR, exist_ok=True)
        for n, s in enumerate(stems, 1):
            outp = os.path.join(PROP_DIR, s + ".npz")
            if os.path.isfile(outp):
                continue
            img = cv2.imread(os.path.join(IMG_DIR, s + ".jpg"))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            spec = specimen_mask(img)
            keep = propose(P, rgb, spec, topk=args.topk)
            masks = np.stack([k[1] for k in keep]).astype(np.uint8) if keep else \
                np.zeros((0, rgb.shape[0], rgb.shape[1]), np.uint8)
            scores = np.array([k[0] for k in keep], np.float32)
            points = np.array([k[2] for k in keep], np.float32)
            np.savez_compressed(outp, masks=masks, scores=scores, points=points)
            if n % 25 == 0:
                print(f"  [{n}/{len(stems)}] {s[:24]} -> {len(keep)} ứng viên")
        print(f"Xong. Proposals -> {PROP_DIR}/")


if __name__ == "__main__":
    main()
