"""Config-driven RCC semi-auto box-to-mask inference."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.seg_crop_model import SIZE, build_model, make_channels, n_channels, pad_box


def resolve_project_path(path_value: str | os.PathLike[str]) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | os.PathLike[str]) -> dict:
    with open(resolve_project_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_segmenter(config: dict):
    model_cfg = config["model"]
    inf_cfg = config.get("inference", {})
    device_name = inf_cfg.get("device", "auto")
    device = "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    if device == "auto":
        device = "cpu"

    ckpt_path = resolve_project_path(model_cfg["checkpoint"])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", model_cfg["architecture"])
    channels = ckpt.get("channels", model_cfg["channels"])
    net = build_model(arch, n_channels(channels)).to(device)
    net.load_state_dict(ckpt["net"])
    net.eval()
    return net, channels, device


def parse_box(text: str) -> list[float]:
    parts = [float(v) for v in text.replace(";", ",").replace(" ", ",").split(",") if v != ""]
    if len(parts) != 4:
        raise ValueError(f"Box must have four numbers x0,y0,x1,y1, got: {text!r}")
    return parts


def clip_box(box: list[float], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = box
    x0, x1 = sorted((max(0, min(width - 1, int(x0))), max(0, min(width, int(x1)))))
    y0, y1 = sorted((max(0, min(height - 1, int(y0))), max(0, min(height, int(y1)))))
    return [x0, y0, x1, y1]


@torch.no_grad()
def boxes_to_mask(net, mode: str, device: str, bgr: np.ndarray, boxes: list[list[int]], pad: float, threshold: float):
    height, width = bgr.shape[:2]
    union = np.zeros((height, width), dtype=bool)
    for box in boxes:
        x0, y0, x1, y1 = pad_box(list(map(float, box)), pad, width, height)
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(make_channels(np.ascontiguousarray(crop), mode))[None].to(device)
        pred = torch.sigmoid(net(x))[0, 0].float().cpu().numpy()
        pred = cv2.resize(pred, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > threshold
        union[y0:y1, x0:x1] |= pred
    return union


def make_overlay(bgr: np.ndarray, mask: np.ndarray, boxes: list[list[int]]) -> np.ndarray:
    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
    out[mask] = (0.5 * out[mask] + 0.5 * np.array([0, 255, 120])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 0), 4)
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(out, (x0, y0), (x1, y1), (60, 120, 255), 4)
    return out


def resolve_image(image: str) -> Path:
    candidates = [
        Path(image),
        PROJECT_ROOT / image,
        PROJECT_ROOT / "data/raw/images" / image,
        PROJECT_ROOT / "data/raw/images" / f"{image}.jpg",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return Path(image)


def run_one(net, mode: str, device: str, config: dict, image_path: Path, boxes: list[list[float]], out_path: Path, overlay_path: Path | None):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        print(f"[skip] cannot read image: {image_path}")
        return False
    height, width = bgr.shape[:2]
    clipped = [clip_box(box, width, height) for box in boxes]
    inf_cfg = config.get("inference", {})
    mask = boxes_to_mask(
        net,
        mode,
        device,
        bgr,
        clipped,
        float(inf_cfg.get("box_pad_fraction", 0.15)),
        float(inf_cfg.get("threshold", 0.5)),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mask.astype(np.uint8) * int(inf_cfg.get("output_mask_value", 255)))
    if overlay_path:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay = make_overlay(bgr, mask, clipped)
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(
        f"[ok] {image_path.name} boxes={len(clipped)} tumor_px={int(mask.sum())} "
        f"tumor_frac={mask.sum() / max(1, height * width):.4f} -> {out_path}"
    )
    return True


def read_csv_boxes(path: Path) -> dict[str, list[list[float]]]:
    by_image: dict[str, list[list[float]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image = row["image"].strip()
            by_image.setdefault(image, []).append([float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"])])
    return by_image


def main() -> None:
    parser = argparse.ArgumentParser(description="RCC SegResNet+LAB crop box-to-mask inference")
    parser.add_argument("--config", default="configs/model/seg_crop_lab.yaml")
    parser.add_argument("--image")
    parser.add_argument("--box", action="append", default=[])
    parser.add_argument("--out")
    parser.add_argument("--overlay")
    parser.add_argument("--csv")
    parser.add_argument("--out_dir")
    parser.add_argument("--overlay_dir")
    args = parser.parse_args()

    config = load_config(args.config)
    net, mode, device = load_segmenter(config)
    print(f"config={args.config} checkpoint={config['model']['checkpoint']} channels={mode} device={device}")

    if args.csv:
        if not args.out_dir:
            parser.error("--csv requires --out_dir")
        by_image = read_csv_boxes(resolve_project_path(args.csv))
        completed = 0
        for image, boxes in by_image.items():
            image_path = resolve_image(image)
            stem = image_path.stem
            overlay_path = Path(args.overlay_dir) / f"{stem}.jpg" if args.overlay_dir else None
            completed += int(
                run_one(
                    net,
                    mode,
                    device,
                    config,
                    image_path,
                    boxes,
                    resolve_project_path(args.out_dir) / f"{stem}.png",
                    resolve_project_path(overlay_path) if overlay_path else None,
                )
            )
        print(f"done {completed}/{len(by_image)} images")
        return

    if not (args.image and args.box and args.out):
        parser.error("single-image mode requires --image, at least one --box, and --out")
    run_one(
        net,
        mode,
        device,
        config,
        resolve_image(args.image),
        [parse_box(box) for box in args.box],
        resolve_project_path(args.out),
        resolve_project_path(args.overlay) if args.overlay else None,
    )


if __name__ == "__main__":
    main()
