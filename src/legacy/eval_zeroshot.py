"""
Đo Dice ZERO-SHOT của một backbone (đặt qua SAM2_CONFIG/SAM2_CKPT) trên tập test
trong labels/split.json. Dùng để so tiny vs large.

  SAM2_CONFIG=configs/sam2.1_hiera_l SAM2_CKPT=checkpoints/sam2.1_hiera_large.pt \
    python eval_zeroshot.py
"""
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from finetune_sam2 import CONFIG, CKPT, RES, DEVICE, eval_dice   # đọc env
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def main():
    split = json.load(open(os.path.join(ROOT, "labels/split.json")))
    test = split["val"]
    print(f"Backbone: {CKPT} @ {RES} | test {len(test)} ảnh")
    model = build_sam2(CONFIG, CKPT, device=DEVICE,
                       hydra_overrides_extra=[f"++model.image_size={RES}"])
    pred = SAM2ImagePredictor(model)
    d = eval_dice(pred, test)
    print(f"ZERO-SHOT test Dice = {d:.4f}")
    out = {"ckpt": os.path.basename(CKPT), "res": RES, "n_test": len(test),
           "zero_shot_dice": round(d, 4)}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    tag = "large" if "large" in CKPT else ("tiny" if "tiny" in CKPT else "other")
    json.dump(out, open(os.path.join(ROOT, f"results/zeroshot_{tag}.json"), "w"), indent=1)
    print(f"-> results/zeroshot_{tag}.json")


if __name__ == "__main__":
    main()
