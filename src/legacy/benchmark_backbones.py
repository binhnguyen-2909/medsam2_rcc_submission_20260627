"""
Benchmark SAM2.1 vanilla vs MedSAM2_latest cho segment KHỐI U trên ảnh đại thể.

Dùng BOX PROMPT (khớp deliverable box->mask): cùng một box cho cả 2 model,
so mask nào bám khối u sát hơn. Không có ground-truth -> đánh giá ĐỊNH TÍNH
qua grid overlay + báo mask area & self-IoU score của model.

Box prompt theo TỈ LỆ ảnh (x0,y0,x1,y1 in [0,1]) -> bất biến với kích thước ảnh.
Sửa/ thêm ảnh qua --prompts file.json cùng định dạng PROMPTS bên dưới.

Lưu ý: chạy ở image_size=512 (config t512) cho cả 2 -> công bằng về prompt.
SAM2.1 gốc vốn 1024; nếu SAM2.1 thắng nên test lại ở 1024 cho con số cuối.

  python benchmark_backbones.py
"""
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CONFIG = "configs/sam2.1_hiera_t512"
CKPTS = {
    "SAM2.1": "checkpoints/sam2.1_hiera_tiny.pt",
    "MedSAM2": "checkpoints/MedSAM2_latest.pt",
}
IMG_DIR = "data/20241212"
# box prompt theo tỉ lệ (ước lượng tay; chỉnh lại nếu lệch khối u)
PROMPTS = {
    "SS21-33569^2021_06_01_09_14_18": [0.34, 0.40, 0.49, 0.58],
    "SS21-34107^2021_06_03_09_04_34": [0.34, 0.41, 0.62, 0.73],
    "SS21-35590^2021_06_09_10_24_16": [0.11, 0.10, 0.31, 0.46],
}
OUT_DIR = "results"


def overlay(img_rgb, mask, box_abs, color=(0, 230, 100), title=""):
    out = img_rgb.copy()
    if mask is not None:
        sel = mask.astype(bool)
        out[sel] = (0.55 * out[sel] + 0.45 * np.array(color)).astype(np.uint8)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 0), 5)
    x0, y0, x1, y1 = box_abs
    cv2.rectangle(out, (x0, y0), (x1, y1), (60, 120, 255), 5)
    if title:
        cv2.putText(out, title, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                    (0, 0, 0), 10, cv2.LINE_AA)
        cv2.putText(out, title, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                    (255, 255, 255), 4, cv2.LINE_AA)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=None, help="JSON {stem: [x0,y0,x1,y1] frac}")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--res", type=int, default=512,
                    help="image_size cho model (SAM2.1 gốc=1024; FLARE=512)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    prompts = json.load(open(args.prompts)) if args.prompts else PROMPTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # nạp ảnh + box tuyệt đối
    from glob import glob
    imgs = {}
    for stem, fb in prompts.items():
        p = os.path.join(IMG_DIR, stem + ".jpg")
        if not os.path.isfile(p):  # khớp linh hoạt đuôi '^^' v.v.
            cands = glob(os.path.join(IMG_DIR, stem + "*.jpg"))
            p = cands[0] if cands else p
        bgr = cv2.imread(p)
        if bgr is None:
            print(f"[skip] không đọc được {p}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        box = np.array([fb[0] * W, fb[1] * H, fb[2] * W, fb[3] * H], dtype=np.float32)
        imgs[stem] = {"rgb": rgb, "box": box,
                      "box_int": [int(v) for v in box]}

    results = {}   # stem -> {model: (mask, score, area)}
    rows = []
    for mname, ckpt in CKPTS.items():
        if not os.path.isfile(ckpt):
            print(f"[bỏ] thiếu checkpoint {ckpt}")
            continue
        print(f"\n=== Nạp {mname}: {ckpt} (res={args.res}) ===")
        model = build_sam2(CONFIG, ckpt, device=device,
                           hydra_overrides_extra=[f"++model.image_size={args.res}"])
        predictor = SAM2ImagePredictor(model)
        ac = (torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda"
              else torch.autocast("cpu", enabled=False))
        for stem, d in imgs.items():
            with torch.inference_mode(), ac:
                predictor.set_image(d["rgb"])
                masks, scores, _ = predictor.predict(box=d["box"], multimask_output=True)
            bi = int(np.argmax(scores))
            mask = masks[bi].astype(np.uint8)
            area = int(mask.sum())
            results.setdefault(stem, {})[mname] = (mask, float(scores[bi]), area)
            rows.append({"image": stem, "model": mname,
                         "score": round(float(scores[bi]), 4), "mask_area_px": area})
            print(f"  {stem[:24]:24s} score={scores[bi]:.3f} area={area}")
        del model, predictor
        torch.cuda.empty_cache()

    # grid: mỗi hàng = 1 ảnh; cột = [orig+box | SAM2.1 | MedSAM2]
    model_names = [m for m in CKPTS if m in {r["model"] for r in rows}]
    panels_rows = []
    TW = 760  # bề rộng mỗi panel
    for stem, d in imgs.items():
        cells = [overlay(d["rgb"], None, d["box_int"], title="orig+box")]
        for mname in model_names:
            mask, score, area = results[stem][mname]
            cells.append(overlay(d["rgb"], mask, d["box_int"],
                                  title=f"{mname} s={score:.2f}"))
        cells = [cv2.resize(c, (TW, int(TW * c.shape[0] / c.shape[1]))) for c in cells]
        h = max(c.shape[0] for c in cells)
        cells = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 8, cv2.BORDER_CONSTANT,
                                    value=(30, 30, 30)) for c in cells]
        panels_rows.append(np.concatenate(cells, axis=1))
    w = max(r.shape[1] for r in panels_rows)
    panels_rows = [cv2.copyMakeBorder(r, 8, 8, 0, w - r.shape[1], cv2.BORDER_CONSTANT,
                                      value=(30, 30, 30)) for r in panels_rows]
    grid = np.concatenate(panels_rows, axis=0)
    grid_path = os.path.join(args.out, f"backbone_benchmark_{args.res}.png")
    cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    csv_path = os.path.join(args.out, f"backbone_benchmark_{args.res}.csv")
    with open(csv_path, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=["image", "model", "score", "mask_area_px"])
        w_.writeheader(); w_.writerows(rows)
    print(f"\nGrid -> {grid_path}\nCSV  -> {csv_path}")


if __name__ == "__main__":
    main()
