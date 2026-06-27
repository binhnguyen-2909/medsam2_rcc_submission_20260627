"""
So Dice box->mask của các backbone trên 12 ảnh TEST cố định (labels/split.json val).
Dùng để quyết có đổi backbone (dùng GPU nhiều hơn) cho deliverable không.

Lưu ý: mask GT do SAM2.1-tiny hỗ trợ tạo -> Dice hơi thiên vị tiny; đọc tương đối.
  python bench_backbone_test.py
"""
import json
import os
import time

import numpy as np
import torch

from finetune_sam2 import (build_sam2, SAM2ImagePredictor, eval_dice, ROOT, RES)

BACKBONES = [
    ("tiny", "configs/sam2.1_hiera_t512", "checkpoints/sam2.1_hiera_tiny.pt"),
    ("large", "configs/sam2.1_hiera_l", "checkpoints/sam2.1_hiera_large.pt"),
]


def main():
    split = json.load(open(os.path.join(ROOT, "labels/split.json")))
    test = split["val"]
    print(f"Test cố định: {len(test)} ảnh\n")
    rows = []
    for name, cfg, ckpt in BACKBONES:
        if not os.path.isfile(os.path.join(ROOT, ckpt)):
            print(f"[bỏ] {name}: chưa có {ckpt}")
            continue
        free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"[{name}] free GPU {free:.1f}GB | nạp {cfg} ...")
        model = build_sam2(cfg, ckpt, device="cuda",
                           hydra_overrides_extra=[f"++model.image_size={RES}"])
        pred = SAM2ImagePredictor(model)
        t0 = time.time()
        d = eval_dice(pred, test)
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        print(f"[{name}] test Dice = {d:.4f} | {dt/len(test):.2f}s/ảnh | peak {peak:.1f}GB\n")
        rows.append((name, d, dt / len(test), peak))
        del model, pred
        torch.cuda.empty_cache()

    print("=== TỔNG ===")
    for name, d, spi, peak in rows:
        print(f"  {name:6s} Dice={d:.4f}  {spi:.2f}s/ảnh  peak~{peak:.1f}GB")
    if len(rows) == 2:
        diff = rows[1][1] - rows[0][1]
        print(f"\nlarge - tiny = {diff:+.4f} Dice "
              f"({'large tốt hơn' if diff > 0.005 else 'không khác đáng kể' if abs(diff)<=0.005 else 'tiny tốt hơn'})")


if __name__ == "__main__":
    main()
