"""Evaluate component-aware specimen-strict inference.

This tests the slice-aware idea on the 50 hand-drawn masks and writes demo
overlays for multi-slice SS21-38576 cases.
"""
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from component_strict import clip_mask_to_component, decode_detections_components, specimen_components  # noqa: E402
from detector import DenseDetector  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
from specimen_clean import clean_specimen  # noqa: E402
from specimen_strict import decode_detections_specimen, specimen_post_mask  # noqa: E402


CFG = "configs/sam2.1_hiera_t512"
SAM_CKPT = "checkpoints/sam2.1_hiera_tiny.pt"
DET_CKPT = "checkpoints/detector_recall.pt"
RES = 1024
DEVICE = "cuda"
HMASK = ROOT / "labels_handdraw" / "masks"
IMG_DIR = ROOT / "data" / "20241212"
DEMO_PREFIX = "SS21-38576"


CONFIGS = [
    {"name": "strict050", "mode": "strict", "thr": 0.5, "fb": 0.35, "frac": 0.50, "shrink": True},
    {"name": "comp4_frac045_fb025", "mode": "component", "thr": 0.5, "fb": 0.25, "frac": 0.45, "max_pc": 4},
    {"name": "comp3_frac050_fb025", "mode": "component", "thr": 0.5, "fb": 0.25, "frac": 0.50, "max_pc": 3},
    {"name": "comp2_frac050_fb035", "mode": "component", "thr": 0.5, "fb": 0.35, "frac": 0.50, "max_pc": 2},
    {"name": "comp1_frac050_fb035", "mode": "component", "thr": 0.5, "fb": 0.35, "frac": 0.50, "max_pc": 1},
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


def sam_union_components(predictor, boxes, comp_ids, comps, shape, ac):
    out = np.zeros(shape, bool)
    with torch.inference_mode(), ac:
        for b, ci in zip(boxes, comp_ids):
            mk, sc, _ = predictor.predict(box=np.array(b, np.float32), multimask_output=True)
            one = mk[int(np.argmax(sc))].astype(bool)
            out |= clip_mask_to_component(one, comps[int(ci)]["mask"])
    return out


def overlay(rgb, mask, boxes, comps, max_w=900):
    out = rgb.copy()
    if mask.any():
        out[mask] = (0.45 * out[mask] + 0.55 * np.array([255, 35, 35])).astype(np.uint8)
    for c in comps:
        x0, y0, x1, y1 = c["bbox"]
        cv2.rectangle(out, (x0, y0), (x1, y1), (40, 220, 40), max(2, out.shape[1] // 900))
    for b in boxes:
        cv2.rectangle(
            out,
            (int(b[0]), int(b[1])),
            (int(b[2]), int(b[3])),
            (40, 120, 255),
            max(2, out.shape[1] // 800),
        )
    if out.shape[1] > max_w:
        hh = int(round(max_w * out.shape[0] / out.shape[1]))
        out = cv2.resize(out, (max_w, hh), interpolation=cv2.INTER_AREA)
    return out


def infer_cfg(cfg, predictor, obj, det_boxes, h, w, spec, ac):
    if cfg["mode"] == "strict":
        bx = decode_detections_specimen(
            obj[0].float(),
            det_boxes[0].float(),
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
        comps = specimen_components(spec)
        return bx, mask, comps, np.zeros((len(bx),), np.int32)

    bx, comp_ids, comps = decode_detections_components(
        obj[0].float(),
        det_boxes[0].float(),
        h,
        w,
        spec,
        thr=cfg["thr"],
        fallback_thr=cfg["fb"],
        min_box_spec_frac=cfg["frac"],
        max_box_per_component=cfg["max_pc"],
    )
    mask = sam_union_components(predictor, bx, comp_ids, comps, (h, w), ac)
    return bx, mask, comps, comp_ids


def main():
    out_dir = ROOT / "results" / "component_strict_eval"
    demo_dir = out_dir / "demo_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)

    ac = torch.autocast("cuda", dtype=torch.bfloat16)
    model = build_sam2(CFG, SAM_CKPT, device=DEVICE, hydra_overrides_extra=[f"++model.image_size={RES}"])
    predictor = SAM2ImagePredictor(model)
    ck = torch.load(DET_CKPT, weights_only=False)
    det = DenseDetector(grid=ck.get("grid", 64)).to(DEVICE)
    det.load_state_dict(ck["det"])
    det.eval()

    stems = json.load(open(ROOT / "labels_handdraw" / "select.json"))["stems"]
    have = [s for s in stems if (HMASK / f"{s}.png").is_file()]
    demo_stems = sorted(p.stem for p in IMG_DIR.glob(f"{DEMO_PREFIX}*.jpg"))
    eval_stems = list(dict.fromkeys(have + demo_stems))
    print(f"[eval] handdraw={len(have)} demo={len(demo_stems)} total={len(eval_stems)}", flush=True)

    rows = []
    for si, stem in enumerate(eval_stems, 1):
        ip = IMG_DIR / f"{stem}.jpg"
        bgr = cv2.imread(str(ip))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = bgr.shape[:2]
        gt_path = HMASK / f"{stem}.png"
        gt = cv2.imread(str(gt_path), 0) > 127 if gt_path.is_file() else None
        spec, _, _ = clean_specimen(bgr)
        with torch.inference_mode(), ac:
            predictor.set_image(rgb)
            feat = predictor._features["image_embed"].float()
            obj, det_boxes = det(feat)
        for cfg in CONFIGS:
            bx, mask, comps, comp_ids = infer_cfg(cfg, predictor, obj, det_boxes, h, w, spec, ac)
            outside = float((mask & (spec == 0)).sum() / max(1, mask.sum()))
            row = {
                "config": cfg["name"],
                "stem": stem,
                "has_gt": int(gt is not None),
                "gt_components": "" if gt is None else ncomp(gt),
                "spec_components": len(comps),
                "n_box": int(len(bx)),
                "dice": "" if gt is None else round(dice(mask, gt), 4),
                "mask_over_spec": round(float(mask.sum()) / max(1, spec.sum()), 4),
                "outside_spec_frac": round(outside, 4),
                "boxes_json": json.dumps([[round(float(v), 2) for v in b] for b in bx]),
                "comp_ids_json": json.dumps([int(x) for x in comp_ids]),
            }
            rows.append(row)
            if stem.startswith(DEMO_PREFIX):
                ov = overlay(rgb, mask, bx, comps)
                cv2.imwrite(
                    str(demo_dir / f"{stem}__{cfg['name']}.jpg"),
                    cv2.cvtColor(ov, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
        if si % 5 == 0 or si == 1 or stem.startswith(DEMO_PREFIX):
            print(f"[{si}/{len(eval_stems)}] {stem[:42]}", flush=True)

    csv_path = out_dir / "component_strict_eval.csv"
    with csv_path.open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)

    eval_rows = [r for r in rows if r["has_gt"]]
    summary = []
    for cfg in CONFIGS:
        rr = [r for r in eval_rows if r["config"] == cfg["name"]]
        d = np.asarray([float(r["dice"]) for r in rr], float)
        ng = np.asarray([int(r["gt_components"]) for r in rr], int)
        summary.append({
            "config": cfg["name"],
            "median": float(np.median(d)),
            "mean": float(np.mean(d)),
            "one_u": float(np.median(d[ng <= 1])),
            "multi_u": float(np.median(d[ng > 1])),
            "box_median": float(np.median([int(r["n_box"]) for r in rr])),
            "mask_over_spec_median": float(np.median([float(r["mask_over_spec"]) for r in rr])),
        })
    summary.sort(key=lambda r: (r["median"], r["mean"]), reverse=True)
    json_path = out_dir / "component_strict_summary.json"
    json.dump(summary, json_path.open("w"), indent=1)

    print("\n===== COMPONENT STRICT SUMMARY =====", flush=True)
    for r in summary:
        print(
            f"{r['config']:22s} median={r['median']:.4f} mean={r['mean']:.4f} "
            f"1u={r['one_u']:.4f} >1u={r['multi_u']:.4f} "
            f"box_med={r['box_median']:.0f} mask/spec={r['mask_over_spec_median']:.3f}",
            flush=True,
        )
    print(f"-> {csv_path}", flush=True)
    print(f"-> {json_path}", flush=True)
    print(f"-> {demo_dir}", flush=True)


if __name__ == "__main__":
    main()
