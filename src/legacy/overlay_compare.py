"""
So sánh TRỰC QUAN tiny vs large trên 12 ảnh test: với cùng box GT, vẽ
  - GT (mask người duyệt)         : viền VÀNG
  - tiny box->mask                : viền XANH LÁ
  - large box->mask               : viền ĐỎ
-> results/tiny_vs_large.jpg  (mắt thường tự đánh giá, bỏ qua Dice thiên vị).

Nạp 1 model tại 1 thời điểm (tránh OOM khi GPU free ít).
  python overlay_compare.py
"""
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
from finetune_sam2 import read_rgb, read_mask, components, bbox_of, RES, DEVICE

AC = torch.autocast("cuda", dtype=torch.bfloat16)
BACKBONES = {
    "tiny": ("configs/sam2.1_hiera_t512", "checkpoints/sam2.1_hiera_tiny.pt"),
    "large": ("configs/sam2.1_hiera_l", "checkpoints/sam2.1_hiera_large.pt"),
}


def predict_all(cfg, ckpt, stems):
    """Trả {stem: [pred_mask theo từng cụm GT]}."""
    model = build_sam2(cfg, ckpt, device=DEVICE,
                       hydra_overrides_extra=[f"++model.image_size={RES}"])
    P = SAM2ImagePredictor(model)
    out = {}
    for s in stems:
        rgb = read_rgb(s)
        gt = read_mask(s)
        with torch.inference_mode(), AC:
            P.set_image(rgb)
        preds = []
        for c in components(gt):
            box = np.array(bbox_of(c), dtype=np.float32)
            with torch.inference_mode(), AC:
                m, sc, _ = P.predict(box=box, multimask_output=False)
            preds.append(m[0].astype(bool))
        out[s] = preds
    del model, P
    torch.cuda.empty_cache()
    return out


def contour(img, mask, color, th=3):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, color, th)


def main():
    split = json.load(open(os.path.join(ROOT, "labels/split.json")))
    stems = split["val"]
    print(f"So sánh trên {len(stems)} ảnh test...")
    print("Nạp tiny..."); pt = predict_all(*BACKBONES["tiny"], stems)
    print("Nạp large..."); pl = predict_all(*BACKBONES["large"], stems)

    tiles = []
    for s in stems:
        img = cv2.cvtColor(read_rgb(s), cv2.COLOR_RGB2BGR)
        gt = read_mask(s)
        for c in components(gt):
            contour(img, c, (0, 255, 255), 4)        # GT vàng
        for m in pt[s]:
            contour(img, m, (0, 230, 0), 3)          # tiny xanh lá
        for m in pl[s]:
            contour(img, m, (40, 40, 255), 3)        # large đỏ
        img = cv2.resize(img, (560, 373))
        cv2.putText(img, s[:22], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(img)
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros((373, 560, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i+cols]) for i in range(0, len(tiles), cols)])
    # chú thích màu
    cv2.putText(grid, "vang=GT  xanh=tiny  do=large", (10, grid.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    cv2.imwrite(os.path.join(ROOT, "results/tiny_vs_large.jpg"), grid)
    print("-> results/tiny_vs_large.jpg")


if __name__ == "__main__":
    main()
