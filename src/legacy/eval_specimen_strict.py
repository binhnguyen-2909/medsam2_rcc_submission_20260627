"""Evaluate specimen-strict full-auto inference on 50 hand-drawn masks."""
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from detector import DenseDetector, propose_boxes  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
from specimen_clean import clean_specimen  # noqa: E402
from specimen_strict import decode_detections_specimen, specimen_post_mask  # noqa: E402


CFG = "configs/sam2.1_hiera_t512"
SAM_CKPT = "checkpoints/sam2.1_hiera_tiny.pt"
DET_CKPT = "checkpoints/detector_recall.pt"
RES = 1024
DEVICE = "cuda"
HMASK = "labels_handdraw/masks"
CASE = "SS21-34460^2021_06_03_15_56_55^^"


CONFIGS = [
    {"name": "baseline", "mode": "baseline"},
    {"name": "baseline_clip_spec", "mode": "baseline_clip"},
    {"name": "strict035_shrink", "mode": "strict", "thr": 0.5, "fb": 0.35, "frac": 0.35, "shrink": True},
    {"name": "strict050_shrink", "mode": "strict", "thr": 0.5, "fb": 0.35, "frac": 0.50, "shrink": True},
    {"name": "strict065_shrink", "mode": "strict", "thr": 0.5, "fb": 0.35, "frac": 0.65, "shrink": True},
    {"name": "strict050_no_shrink", "mode": "strict", "thr": 0.5, "fb": 0.35, "frac": 0.50, "shrink": False},
    {"name": "strict050_thr04", "mode": "strict", "thr": 0.4, "fb": 0.25, "frac": 0.50, "shrink": True},
    {"name": "strict050_thr06", "mode": "strict", "thr": 0.6, "fb": 0.35, "frac": 0.50, "shrink": True},
]


def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def ncomp(m):
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return max(1, int((st[1:, cv2.CC_STAT_AREA] >= 0.002 * m.size).sum()))


def sam_union(predictor, boxes, shape, ac):
    out = np.zeros(shape, bool)
    with torch.inference_mode(), ac:
        for b in boxes:
            mk, sc, _ = predictor.predict(box=np.array(b, np.float32), multimask_output=True)
            out |= mk[int(np.argmax(sc))].astype(bool)
    return out


def main():
    os.makedirs("results", exist_ok=True)
    ac = torch.autocast("cuda", dtype=torch.bfloat16)
    model = build_sam2(CFG, SAM_CKPT, device=DEVICE, hydra_overrides_extra=[f"++model.image_size={RES}"])
    predictor = SAM2ImagePredictor(model)
    ck = torch.load(DET_CKPT, weights_only=False)
    det = DenseDetector(grid=ck.get("grid", 64)).to(DEVICE)
    det.load_state_dict(ck["det"])
    det.eval()
    stems = json.load(open("labels_handdraw/select.json"))["stems"]
    have = [s for s in stems if os.path.isfile(f"{HMASK}/{s}.png")]
    if CASE not in have and os.path.isfile(f"data/20241212/{CASE}.jpg"):
        have = [CASE] + have
    print(f"[eval] {len(have)} cases including diagnostic case={CASE}", flush=True)
    rows = []
    for si, s in enumerate(have, 1):
        bgr = cv2.imread(f"data/20241212/{s}.jpg")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = bgr.shape[:2]
        gt_path = f"{HMASK}/{s}.png"
        gt = cv2.imread(gt_path, 0) > 127 if os.path.isfile(gt_path) else None
        spec, _, _ = clean_specimen(bgr)
        with torch.inference_mode(), ac:
            predictor.set_image(rgb)
            feat = predictor._features["image_embed"].float()
            obj, boxes = det(feat)
        for cfg in CONFIGS:
            if cfg["mode"] in ("baseline", "baseline_clip"):
                bx = propose_boxes(obj[0].float(), boxes[0].float(), h, w, spec=spec, thr=0.5)
                mask = sam_union(predictor, bx, (h, w), ac)
                if cfg["mode"] == "baseline_clip":
                    mask = specimen_post_mask(mask, spec)
            else:
                bx = decode_detections_specimen(
                    obj[0].float(),
                    boxes[0].float(),
                    h,
                    w,
                    spec,
                    thr=cfg["thr"],
                    fallback_thr=cfg["fb"],
                    min_box_spec_frac=cfg["frac"],
                    shrink_to_spec=cfg["shrink"],
                )
                mask = sam_union(predictor, bx, (h, w), ac)
                mask = specimen_post_mask(mask, spec)
            outfrac = float((mask & (spec == 0)).sum() / max(1, mask.sum()))
            rows.append({
                "config": cfg["name"],
                "stem": s,
                "has_gt": int(gt is not None),
                "n_obj": "" if gt is None else ncomp(gt),
                "n_box": int(len(bx)),
                "dice": "" if gt is None else round(dice(mask, gt), 4),
                "mask_over_spec": round(float(mask.sum()) / max(1, spec.sum()), 4),
                "outside_spec_frac": round(outfrac, 4),
            })
        if s == CASE:
            print("\n[diagnostic SS21-34460]", flush=True)
            for r in [x for x in rows if x["stem"] == CASE]:
                print(
                    f"  {r['config']:20s} n_box={r['n_box']:2d} "
                    f"mask/spec={r['mask_over_spec']:.3f} outside={r['outside_spec_frac']:.3f}",
                    flush=True,
                )
        if si % 5 == 0 or si == 1:
            print(f"[{si}/{len(have)}] {s[:24]}", flush=True)

    eval_rows = [r for r in rows if r["has_gt"]]
    summary = []
    for cfg in CONFIGS:
        rr = [r for r in eval_rows if r["config"] == cfg["name"]]
        d = np.array([float(r["dice"]) for r in rr], float)
        no = np.array([int(r["n_obj"]) for r in rr], int)
        summary.append({
            "config": cfg["name"],
            "median": float(np.median(d)),
            "mean": float(np.mean(d)),
            "one_u": float(np.median(d[no <= 1])),
            "multi_u": float(np.median(d[no > 1])),
            "box_median": float(np.median([r["n_box"] for r in rr])),
            "mask_over_spec_median": float(np.median([r["mask_over_spec"] for r in rr])),
        })
    summary.sort(key=lambda r: (r["median"], r["mean"]), reverse=True)
    csv_path = "results/specimen_strict_eval.csv"
    with open(csv_path, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)
    json_path = "results/specimen_strict_summary.json"
    json.dump(summary, open(json_path, "w"), indent=1)
    print("\n===== SPECIMEN STRICT SUMMARY =====", flush=True)
    for r in summary:
        print(
            f"{r['config']:22s} median={r['median']:.4f} mean={r['mean']:.4f} "
            f"1u={r['one_u']:.4f} >1u={r['multi_u']:.4f} "
            f"box_med={r['box_median']:.0f} mask/spec={r['mask_over_spec_median']:.3f}",
            flush=True,
        )
    print(f"-> {csv_path}", flush=True)
    print(f"-> {json_path}", flush=True)


if __name__ == "__main__":
    main()
