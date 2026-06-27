"""Batch full-auto inference on RCC gross pathology images.

Pipeline: SAM2.1 tiny image encoder -> detector_recall dense boxes -> SAM box masks.
Outputs full-resolution binary masks, thumbnail overlays, and CSV summaries.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from detector import DenseDetector, propose_boxes  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
from specimen_clean import clean_specimen  # noqa: E402


CFG = "configs/sam2.1_hiera_t512"
SAM_CKPT = "checkpoints/sam2.1_hiera_tiny.pt"
DET_CKPT = "checkpoints/detector_recall.pt"
RES = 1024
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def read_mask(path, shape):
    if not path.is_file():
        return None
    m = cv2.imread(str(path), 0)
    if m is None:
        return None
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def overlay_thumb(rgb, mask, boxes, max_w):
    out = rgb.copy()
    if mask.any():
        out[mask] = (0.45 * out[mask] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
    for b in boxes:
        cv2.rectangle(
            out,
            (int(b[0]), int(b[1])),
            (int(b[2]), int(b[3])),
            (60, 120, 255),
            max(2, int(round(out.shape[1] / 900))),
        )
    if out.shape[1] > max_w:
        h = int(round(max_w * out.shape[0] / out.shape[1]))
        out = cv2.resize(out, (max_w, h), interpolation=cv2.INTER_AREA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default="data/20241212")
    ap.add_argument("--out", default="results/full_auto_all_detector_recall_20260626")
    ap.add_argument("--det_ckpt", default=DET_CKPT)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--max", type=int, default=0, help="0 means all images")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--thumb_w", type=int, default=900)
    ap.add_argument("--no_overlays", action="store_true")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out)
    mask_dir = out_dir / "masks"
    ov_dir = out_dir / "overlays_thumb"
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_overlays:
        ov_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.max > 0:
        images = images[: args.max]
    if not images:
        raise SystemExit(f"No images found in {img_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("CUDA is required by this pipeline in the current repo scripts.")
    ac = torch.autocast("cuda", dtype=torch.bfloat16)

    print(f"[load] SAM={SAM_CKPT} detector={args.det_ckpt} thr={args.thr}", flush=True)
    sam = build_sam2(CFG, SAM_CKPT, device=device, hydra_overrides_extra=[f"++model.image_size={RES}"])
    predictor = SAM2ImagePredictor(sam)
    ck = torch.load(args.det_ckpt, weights_only=False)
    det = DenseDetector(grid=ck.get("grid", 64)).to(device)
    det.load_state_dict(ck["det"])
    det.eval()
    print(f"[detector] epoch={ck.get('epoch')} grid={ck.get('grid', 64)} val_dice={ck.get('val_dice')}", flush=True)
    print(f"[run] images={len(images)} -> {out_dir}", flush=True)

    summary_rows = []
    box_rows = []
    t0 = time.time()
    for i, ip in enumerate(images, 1):
        stem = ip.stem
        mask_path = mask_dir / f"{stem}.png"
        ov_path = ov_dir / f"{stem}.jpg"
        if mask_path.exists() and not args.overwrite:
            if i % 25 == 0 or i == len(images):
                print(f"[{i}/{len(images)}] skip existing {stem}", flush=True)
            continue

        bgr = cv2.imread(str(ip))
        if bgr is None:
            print(f"[warn] cannot read {ip}", flush=True)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        spec, _, _ = clean_specimen(bgr)

        with torch.inference_mode(), ac:
            predictor.set_image(rgb)
            feat = predictor._features["image_embed"].float()
            obj, boxes = det(feat)
            boxes_px = propose_boxes(obj[0].float(), boxes[0].float(), h, w, spec=spec, thr=args.thr)
            union = np.zeros((h, w), bool)
            for b in boxes_px:
                mk, sc, _ = predictor.predict(box=b, multimask_output=True)
                union |= mk[int(np.argmax(sc))].astype(bool)

        cv2.imwrite(str(mask_path), (union.astype(np.uint8) * 255), [cv2.IMWRITE_PNG_COMPRESSION, 4])
        if not args.no_overlays:
            ov = overlay_thumb(rgb, union, boxes_px, args.thumb_w)
            cv2.imwrite(str(ov_path), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])

        sam_gt = read_mask(ROOT / "labels" / "masks" / f"{stem}.png", (h, w))
        hand_gt = read_mask(ROOT / "labels_handdraw" / "masks" / f"{stem}.png", (h, w))
        mask_area = int(union.sum())
        spec_area = int((spec > 0).sum())
        boxes_list = [[round(float(v), 2) for v in b] for b in boxes_px]
        row = {
            "stem": stem,
            "image": str(ip),
            "mask": str(mask_path),
            "overlay_thumb": "" if args.no_overlays else str(ov_path),
            "width": w,
            "height": h,
            "n_box": int(len(boxes_px)),
            "mask_area_px": mask_area,
            "mask_frac": round(mask_area / max(1, h * w), 6),
            "specimen_area_px": spec_area,
            "mask_over_specimen": round(mask_area / max(1, spec_area), 6),
            "dice_vs_sam_label": "" if sam_gt is None else round(dice(union, sam_gt), 4),
            "dice_vs_handdraw": "" if hand_gt is None else round(dice(union, hand_gt), 4),
            "boxes_json": json.dumps(boxes_list, ensure_ascii=True),
        }
        summary_rows.append(row)
        for j, b in enumerate(boxes_list):
            box_rows.append({
                "stem": stem,
                "box_idx": j,
                "x0": b[0],
                "y0": b[1],
                "x1": b[2],
                "y1": b[3],
                "box_area_px": round(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]), 2),
            })

        if i % 10 == 0 or i == 1 or i == len(images):
            dt = time.time() - t0
            print(
                f"[{i}/{len(images)}] {stem} boxes={len(boxes_px)} "
                f"mask/spec={row['mask_over_specimen']:.3f} elapsed={dt/60:.1f}m",
                flush=True,
            )

    summary_path = out_dir / "summary.csv"
    boxes_path = out_dir / "boxes.csv"
    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            wri.writeheader()
            wri.writerows(summary_rows)
    else:
        summary_path.touch()
    with open(boxes_path, "w", newline="") as f:
        fields = ["stem", "box_idx", "x0", "y0", "x1", "y1", "box_area_px"]
        wri = csv.DictWriter(f, fieldnames=fields)
        wri.writeheader()
        wri.writerows(box_rows)

    if summary_rows:
        n_box = np.array([r["n_box"] for r in summary_rows], float)
        mfrac = np.array([r["mask_over_specimen"] for r in summary_rows], float)
        print("\n===== FULL-AUTO ALL SUMMARY =====", flush=True)
        print(f"processed={len(summary_rows)} / requested={len(images)}", flush=True)
        print(f"box/image median={np.median(n_box):.0f} mean={np.mean(n_box):.2f}", flush=True)
        print(f"mask/specimen median={np.median(mfrac):.4f} mean={np.mean(mfrac):.4f}", flush=True)
        print(f"-> {summary_path}", flush=True)
        print(f"-> {boxes_path}", flush=True)
        print(f"-> {mask_dir}", flush=True)
        if not args.no_overlays:
            print(f"-> {ov_dir}", flush=True)


if __name__ == "__main__":
    main()
