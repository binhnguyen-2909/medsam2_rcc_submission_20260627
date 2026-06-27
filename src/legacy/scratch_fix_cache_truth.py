"""
Vá CACHE (CPU, KHÔNG GPU) cho các stem có mask THẬT mới vẽ: feature SAM phụ thuộc ẢNH
(không đổi) nên GIỮ NGUYÊN; chỉ tính lại box+comp GT từ mask thật. Tránh rebuild SAM@1024
(tốn GPU đang kẹt). Sau khi chạy, mtime cache > mtime mask -> train_detector không rebuild.
Chạy: /home/hvusynh2/conda_envs/medsam2_anno/bin/python scratch_fix_cache_truth.py
"""
import os, sys, json, numpy as np, torch
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from train_detector import gt_targets, cache_path   # gt_targets dùng mask_path -> ưu tiên truth

def main():
    done = json.load(open("labels_truth/done.json"))
    n_fix = n_skip = 0
    for s in done:
        gt = gt_targets(s)   # đọc mask THẬT -> (boxes, comps, hw)
        if gt is None:
            print(f"  {s[:22]} mask rỗng/không hợp lệ — bỏ"); n_skip += 1; continue
        boxes, comps, hw = gt
        for fl in (False, True):
            cp = cache_path(s, fl)
            if not os.path.isfile(cp): continue
            ck = torch.load(cp, weights_only=False)   # giữ feat/hrf
            b = boxes.copy(); c = comps.copy()
            if fl:
                c = c[:, :, ::-1].copy(); b = b.copy(); b[:, 0] = 1.0 - b[:, 0]
            ck["boxes"] = b; ck["comps"] = (c > 0.5).astype(np.uint8) if c.dtype != np.uint8 else c
            ck["hw"] = hw
            torch.save(ck, cp)
            n_fix += 1
    print(f"\nVá xong {n_fix} cache (CPU, giữ feature) cho {len(done)} stem thật; bỏ {n_skip}.", flush=True)

if __name__ == "__main__":
    main()
