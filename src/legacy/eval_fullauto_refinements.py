"""Evaluate cheap full-auto refinements on the 50 hand-drawn masks.

The baseline is the current champion: detector_recall -> SAM2.1 tiny.
This script tests post-processing and detector decode knobs before running
anything on the full image set.
"""
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch
from skimage.segmentation import inverse_gaussian_gradient
from skimage.segmentation import morphological_geodesic_active_contour as MGAC

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from detector import DenseDetector, cxcywh_to_xyxy, decode_detections, propose_boxes  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
from specimen_clean import clean_specimen  # noqa: E402


CFG = "configs/sam2.1_hiera_t512"
SAM_CKPT = "checkpoints/sam2.1_hiera_tiny.pt"
DET_CKPT = "checkpoints/detector_recall.pt"
RES = 1024
DEVICE = "cuda"
HMASK = "labels_handdraw/masks"


def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def ncomp(m):
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return max(1, int((st[1:, cv2.CC_STAT_AREA] >= 0.002 * m.size).sum()))


def decode_custom(obj_logits, boxes, h, w, spec, thr=0.5, nms_iou=0.5, max_box=20):
    bxk, _ = decode_detections(obj_logits, boxes, thr=thr, nms_iou=nms_iou, max_box=max_box)
    xyxy = cxcywh_to_xyxy(bxk).clamp(0, 1).cpu().numpy()
    px = (xyxy * np.array([w, h, w, h], np.float32)).astype(np.float32)
    if spec is None or spec.sum() == 0 or len(px) == 0:
        return px
    keep = []
    for b in px:
        cx = int(np.clip((b[0] + b[2]) / 2, 0, w - 1))
        cy = int(np.clip((b[1] + b[3]) / 2, 0, h - 1))
        keep.append(spec[cy, cx] > 0)
    return px[np.array(keep, bool)]


def sam_union(predictor, boxes, shape, ac):
    out = np.zeros(shape, bool)
    with torch.inference_mode(), ac:
        for b in boxes:
            mk, sc, _ = predictor.predict(box=np.array(b, np.float32), multimask_output=True)
            out |= mk[int(np.argmax(sc))].astype(bool)
    return out


def keep_largest(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    areas = st[1:, cv2.CC_STAT_AREA]
    i = int(np.argmax(areas)) + 1
    return lab == i


def remove_small(mask, min_frac=0.001):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask, bool)
    thr = min_frac * mask.size
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= thr:
            out |= lab == i
    return out


def erode_open(mask, erode_k=3, open_k=5):
    if mask.sum() == 0:
        return mask
    eker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    oker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    m = cv2.erode(mask.astype(np.uint8), eker, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, oker)
    return m > 0


def active_contour(mask, bgr, spec):
    if mask.sum() == 0:
        return mask
    ys, xs = np.where(mask)
    pad = 80
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad)
    sub = bgr[y0:y1, x0:x1]
    init = mask[y0:y1, x0:x1].astype(np.uint8)
    if init.sum() == 0:
        return mask
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    edge = inverse_gaussian_gradient(gray, alpha=100, sigma=3)
    try:
        ref = MGAC(edge, num_iter=10, init_level_set=init, smoothing=2, balloon=-1) > 0
    except Exception:
        return mask
    out = np.zeros_like(mask, bool)
    out[y0:y1, x0:x1] = ref
    if spec is not None and spec.sum() > 0:
        out &= spec > 0
    return out


def apply_post(mask, bgr, spec, steps):
    out = mask.copy()
    for step in steps:
        if step == "specimen":
            out &= spec > 0
        elif step == "mincc":
            out = remove_small(out)
        elif step == "largest":
            out = keep_largest(out)
        elif step == "erode_open":
            out = erode_open(out)
        elif step == "active":
            out = active_contour(out, bgr, spec)
    return out


CONFIGS = [
    {"name": "baseline", "decode": "official", "thr": 0.5, "nms": 0.5, "steps": []},
    {"name": "color_specimen", "decode": "official", "thr": 0.5, "nms": 0.5, "steps": ["specimen"]},
    {"name": "specimen_mincc", "decode": "official", "thr": 0.5, "nms": 0.5, "steps": ["specimen", "mincc"]},
    {"name": "specimen_largest", "decode": "official", "thr": 0.5, "nms": 0.5, "steps": ["specimen", "largest"]},
    {"name": "specimen_erode_open", "decode": "official", "thr": 0.5, "nms": 0.5, "steps": ["specimen", "erode_open"]},
    {"name": "specimen_active", "decode": "official", "thr": 0.5, "nms": 0.5, "steps": ["specimen", "active"]},
    {"name": "thr06_specimen", "decode": "custom", "thr": 0.6, "nms": 0.5, "steps": ["specimen"]},
    {"name": "thr07_specimen", "decode": "custom", "thr": 0.7, "nms": 0.5, "steps": ["specimen"]},
    {"name": "nms03_specimen", "decode": "custom", "thr": 0.5, "nms": 0.3, "steps": ["specimen"]},
    {"name": "nms07_specimen", "decode": "custom", "thr": 0.5, "nms": 0.7, "steps": ["specimen"]},
    {"name": "thr06_nms03_specimen", "decode": "custom", "thr": 0.6, "nms": 0.3, "steps": ["specimen"]},
    {"name": "thr06_nms03_erode_open", "decode": "custom", "thr": 0.6, "nms": 0.3, "steps": ["specimen", "erode_open"]},
]


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
    print(f"[eval] {len(have)} handdraw masks | detector={DET_CKPT} ep={ck.get('epoch')}", flush=True)

    rows = []
    for si, s in enumerate(have, 1):
        gt = cv2.imread(f"{HMASK}/{s}.png", 0) > 127
        bgr = cv2.imread(f"data/20241212/{s}.jpg")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = gt.shape
        spec, _, _ = clean_specimen(bgr)
        with torch.inference_mode(), ac:
            predictor.set_image(rgb)
            feat = predictor._features["image_embed"].float()
            obj, boxes = det(feat)
        cache = {}
        for cfg in CONFIGS:
            key = (cfg["decode"], cfg["thr"], cfg["nms"])
            if key not in cache:
                if cfg["decode"] == "official":
                    bx = propose_boxes(obj[0].float(), boxes[0].float(), h, w, spec=spec, thr=cfg["thr"])
                else:
                    bx = decode_custom(obj[0].float(), boxes[0].float(), h, w, spec, cfg["thr"], cfg["nms"])
                cache[key] = (bx, sam_union(predictor, bx, gt.shape, ac))
            bx, raw = cache[key]
            pred = apply_post(raw, bgr, spec, cfg["steps"])
            rows.append({
                "config": cfg["name"],
                "stem": s,
                "n_obj": ncomp(gt),
                "n_box": int(len(bx)),
                "dice": round(dice(pred, gt), 4),
                "mask_px": int(pred.sum()),
            })
        if si % 5 == 0 or si == 1:
            bd = [r for r in rows if r["stem"] == s and r["config"] == "baseline"][0]["dice"]
            print(f"[{si}/{len(have)}] {s[:24]} baseline={bd:.4f}", flush=True)

    csv_path = "results/fullauto_refinement_eval.csv"
    with open(csv_path, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)

    summary = []
    for cfg in CONFIGS:
        rr = [r for r in rows if r["config"] == cfg["name"]]
        d = np.array([r["dice"] for r in rr], float)
        no = np.array([r["n_obj"] for r in rr], int)
        summary.append({
            "config": cfg["name"],
            "median": float(np.median(d)),
            "mean": float(np.mean(d)),
            "one_u_median": float(np.median(d[no <= 1])),
            "multi_u_median": float(np.median(d[no > 1])),
            "box_median": float(np.median([r["n_box"] for r in rr])),
        })
    summary.sort(key=lambda r: (r["median"], r["mean"]), reverse=True)
    json_path = "results/fullauto_refinement_summary.json"
    json.dump(summary, open(json_path, "w"), indent=1)
    print("\n===== FULL-AUTO REFINEMENT SUMMARY =====", flush=True)
    for r in summary:
        print(
            f"{r['config']:26s} median={r['median']:.4f} mean={r['mean']:.4f} "
            f"1u={r['one_u_median']:.4f} >1u={r['multi_u_median']:.4f} box_med={r['box_median']:.0f}",
            flush=True,
        )
    print(f"-> {csv_path}", flush=True)
    print(f"-> {json_path}", flush=True)


if __name__ == "__main__":
    main()
