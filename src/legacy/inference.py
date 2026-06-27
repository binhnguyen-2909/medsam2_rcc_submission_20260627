"""
python inference.py \
    --img_dir   2021_images \
    --checkpoint exp_log/medsam2_2021_scratch/checkpoints/checkpoint_120.pt \
    --config    sam2/configs/medsam2_2021.yaml \
    --result_dir results \
    --grid_cols 4        

"""

import argparse
import json
import os
import sys
from glob import glob
from math import ceil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Make sure the project root is on sys.path ────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import torch

from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_labelme_polygons(json_path: str):
    with open(json_path) as f:
        data = json.load(f)
    polygons = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") == "polygon" and len(shape.get("points", [])) >= 3:
            pts = np.array(shape["points"], dtype=np.float32)  # (N, 2) xy
            polygons.append((shape.get("label", "unknown"), pts))
    return polygons


def polygon_to_bbox(pts: np.ndarray, pad: float = 0.05, img_hw=None):

    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    w, h = x_max - x_min, y_max - y_min
    x_min -= pad * w
    y_min -= pad * h
    x_max += pad * w
    y_max += pad * h
    if img_hw is not None:
        H, W = img_hw
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(W - 1, x_max)
        y_max = min(H - 1, y_max)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def centre_bbox(img_hw, frac: float = 0.4):
    H, W = img_hw
    side = frac * min(H, W)
    cx, cy = W / 2, H / 2
    x_min = max(0, cx - side / 2)
    y_min = max(0, cy - side / 2)
    x_max = min(W - 1, cx + side / 2)
    y_max = min(H - 1, cy + side / 2)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def sliding_window(img_hw, scales=(0.25, 0.40, 0.55, 0.70),
                              stride_frac=0.5):
    H, W = img_hw
    proposals = []
    for frac in scales:
        side_h = frac * H
        side_w = frac * W
        stride_y = max(1, int(side_h * stride_frac))
        stride_x = max(1, int(side_w * stride_frac))
        y0 = 0
        while y0 + side_h <= H:
            x0 = 0
            while x0 + side_w <= W:
                proposals.append(np.array(
                    [x0, y0, x0 + side_w, y0 + side_h], dtype=np.float32
                ))
                x0 += stride_x
            y0 += stride_y
    return proposals


def best_prompt_bbox(predictor: "SAM2ImagePredictor", img_hw):
    proposals = sliding_window(img_hw)
    best_score = -1.0
    best_bbox = centre_bbox(img_hw, frac=0.4)   # safe default
    
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else torch.autocast("cpu", enabled=False)
    
    for bbox in proposals:
        with torch.inference_mode(), autocast_ctx:
            _, scores, _ = predictor.predict(
                box=bbox,
                multimask_output=False,  # fast – just need the score
            )
        score = float(scores.max())
        if score > best_score:
            best_score = score
            best_bbox = bbox
    return best_bbox, best_score


def polygon_to_mask(pts: np.ndarray, H: int, W: int) -> np.ndarray:
    """Rasterise a polygon into a binary mask of shape (H, W)."""
    mask = np.zeros((H, W), dtype=np.uint8)
    pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts_int], 1)
    return mask


def compute_hd95(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree

    pred_bool = pred_mask.astype(bool)
    gt_bool   = gt_mask.astype(bool)

    if not pred_bool.any() and not gt_bool.any():
        return 0.0
    if not pred_bool.any() or not gt_bool.any():
        return np.inf

    def get_boundary(mask):
        eroded = binary_erosion(mask)
        boundary = mask & ~eroded
        pts = np.argwhere(boundary)
        return pts if len(pts) > 0 else np.argwhere(mask)

    pred_pts = get_boundary(pred_bool)
    gt_pts   = get_boundary(gt_bool)

    dist_p2g, _ = cKDTree(gt_pts).query(pred_pts)
    dist_g2p, _ = cKDTree(pred_pts).query(gt_pts)

    return float(np.percentile(np.concatenate([dist_p2g, dist_g2p]), 95))


def draw_metrics_topright(img: np.ndarray, metrics: dict) -> np.ndarray:
    img = img.copy()
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.72
    thickness  = 2
    line_h     = 30
    pad        = 8
    H, W       = img.shape[:2]

    lines = [f"{k}: {v}" for k, v in metrics.items()]

    # Measure widest line to set background rect width
    max_w = max(cv2.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines)
    rect_x0 = W - max_w - pad * 2
    rect_y0 = 0
    rect_x1 = W
    rect_y1 = line_h * len(lines) + pad

    # Semi-transparent dark background
    roi = img[rect_y0:rect_y1, rect_x0:rect_x1]
    dark = (roi * 0.45).astype(np.uint8)
    img[rect_y0:rect_y1, rect_x0:rect_x1] = dark

    for i, line in enumerate(lines):
        tw, th = cv2.getTextSize(line, font, font_scale, thickness)[0]
        x = W - tw - pad
        y = (i + 1) * line_h - pad // 2
        cv2.putText(img, line, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img


def apply_mask_overlay(img_rgb: np.ndarray, mask: np.ndarray,
                       color=(0, 255, 100), alpha=0.45) -> np.ndarray:
    """Blend a binary mask onto an RGB image."""
    overlay = img_rgb.copy()
    overlay[mask.astype(bool)] = (
        (1 - alpha) * overlay[mask.astype(bool)] + alpha * np.array(color)
    ).astype(np.uint8)
    return overlay


def draw_bbox(img_rgb: np.ndarray, bbox: np.ndarray,
              color=(255, 60, 60), thickness=3) -> np.ndarray:
    x0, y0, x1, y1 = bbox.astype(int)
    return cv2.rectangle(img_rgb.copy(), (x0, y0), (x1, y1), color, thickness)


def build_grid(images: list, ncols: int, padding: int = 10,
               bg: tuple = (30, 30, 30)) -> np.ndarray:
    if not images:
        raise ValueError("No images to arrange in grid")

    target_h = max(img.shape[0] for img in images)
    target_w = max(img.shape[1] for img in images)

    def pad_to(img, h, w):
        out = np.full((h, w, 3), bg, dtype=np.uint8)
        out[: img.shape[0], : img.shape[1]] = img
        return out

    images = [pad_to(img, target_h, target_w) for img in images]

    nrows = ceil(len(images) / ncols)
    # Pad to fill the grid
    while len(images) < nrows * ncols:
        images.append(np.full((target_h, target_w, 3), bg, dtype=np.uint8))

    rows = []
    for r in range(nrows):
        row_imgs = images[r * ncols: (r + 1) * ncols]
        row = np.concatenate(
            [img for img in row_imgs], axis=1
        )
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    return grid


def parse_args():
    parser = argparse.ArgumentParser(description="MedSAM2 inference on 2021_images dataset")
    parser.add_argument("--img_dir", type=str, default="2021_images",
                        help="Directory to data")
    parser.add_argument("--checkpoint", type=str,
                        default="exp_log/medsam2_2021/checkpoints/checkpoint.pt",
                        help="Path to fine-tuned MedSAM2 checkpoint")
    parser.add_argument("--config", type=str,
                        default="medsam2_2021",
                        help="Hydra config name in (sam2/configs/)")
    parser.add_argument("--result_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--grid_cols", type=int, default=4,
                        help="Number of columns in the output grid PNG")
    parser.add_argument("--no_json_only", action="store_true",
                        help="Only process images without a JSON annotation file")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of images to process")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda / cpu). Defaults to auto-detect.")
    parser.add_argument("--bbox_pad", type=float, default=0.05,
                        help="Fractional padding added to GT polygon bbox prompt")
    parser.add_argument("--prompt_mode", type=str, default="sliding",
                        choices=["sliding", "centre"],
                        help="Prompt strategy for non-annotated images: "
                             "'sliding'  "
                             "or 'centre'")
    return parser.parse_args()

def load_model(config_name: str, ckpt_path: str, device: str) -> SAM2ImagePredictor:
    """
    Build the SAM2Train model incude wirh Lora
    from the training YAML config, then load the fine-tuned checkpoint weights.

    Use `SAM2Train` creating the lora_A / lora_B
    parameters that must exist before we can load the checkpoint.
    """
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    # The training config lives in sam2/configs/ and has a `trainer.model` section
    # that points to `training.model.sam2.SAM2Train`.
    cfg_path = os.path.join(script_dir, "sam2", "configs", f"{config_name}.yaml")
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(script_dir, "sam2", "configs", config_name)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    print(f"Loading training config from: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    model_cfg = cfg.trainer.model

    # Manually override image_size from the scratch block if present
    if hasattr(cfg, "scratch") and hasattr(cfg.scratch, "resolution"):
        OmegaConf.update(model_cfg, "image_size", cfg.scratch.resolution, merge=True)

    print(f"Instantiating {model_cfg._target_} ...")
    model = instantiate(model_cfg, _recursive_=True)

    # Load the fine-tuned weights
    print(f"Loading fine-tuned weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print("Weights loaded successfully.")

    model = model.to(device).eval()
    predictor = SAM2ImagePredictor(model)
    return predictor


def infer_image(predictor: SAM2ImagePredictor,
                img_path: str,
                json_path: str | None,
                bbox_pad: float,
                device: str,
                prompt_mode: str = "sliding"):
    """
    Run inference on a single image.

    Returns
    -------
    result_img : np.ndarray  HxWx3 RGB  – side-by-side panel (original | overlay)
    pred_mask  : np.ndarray  HxW bool
    pred_bbox  : np.ndarray  [x0,y0,x1,y1]
    gt_mask    : np.ndarray | None  HxW bool (if JSON exists)
    iou        : float | None
    dice       : float | None
    hd95       : float | None
    """
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil)          # H x W x 3, uint8, RGB
    H, W = img_np.shape[:2]

    # ── Determine prompt bbox ────────────────────────────────────────────────
    gt_mask = None
    gt_polygons = []
    if json_path is not None and os.path.exists(json_path):
        gt_polygons = load_labelme_polygons(json_path)

    if gt_polygons:
        # Use all GT polygons; merge their bboxes for the prompt
        all_pts = np.concatenate([pts for _, pts in gt_polygons], axis=0)
        prompt_bbox = polygon_to_bbox(all_pts, pad=bbox_pad, img_hw=(H, W))
        # Build merged binary GT mask for visualisation
        gt_mask = np.zeros((H, W), dtype=np.uint8)
        for _, pts in gt_polygons:
            gt_mask |= polygon_to_mask(pts, H, W)
        gt_mask = gt_mask.astype(bool)
    else:
        
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
        
        with torch.inference_mode(), autocast_ctx:
            predictor.set_image(img_np)

        if prompt_mode == "sliding":
            print(f"  [sliding window] scanning {len(sliding_window((H,W)))} proposals …", end="", flush=True)
            prompt_bbox, best_iou = best_prompt_bbox(predictor, (H, W))
            print(f" best IoU={best_iou:.3f}")
        else:
            prompt_bbox = centre_bbox((H, W), frac=0.4)

    # SAM2 final inference with chosen prompt
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
    with torch.inference_mode(), autocast_ctx:
        if gt_polygons:
            predictor.set_image(img_np)
        masks, scores, _ = predictor.predict(
            box=prompt_bbox,
            multimask_output=True,
        )   # masks: (N_masks, H, W) bool

    # Pick the mask with the highest predicted IoU score
    best_idx = int(np.argmax(scores))
    pred_mask = masks[best_idx]  # (H, W) bool

    # Derive a tight bounding box from the predicted mask for visualisation
    ys, xs = np.where(pred_mask)
    if len(xs) > 0:
        pred_bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
    else:
        pred_bbox = prompt_bbox.copy()

    # ── Visualise ────────────────────────────────────────────────────────────
    orig_panel = img_np.copy()
    overlay_panel = img_np.copy()

    # Draw prompt bbox on both panels (blue)
    orig_panel   = draw_bbox(orig_panel,   prompt_bbox, color=(60, 120, 255), thickness=2)
    overlay_panel = draw_bbox(overlay_panel, prompt_bbox, color=(60, 120, 255), thickness=2)

    # Predicted mask overlay (green) on right panel
    overlay_panel = apply_mask_overlay(overlay_panel, pred_mask,
                                       color=(0, 230, 100), alpha=0.45)
    # Predicted bbox (red)
    overlay_panel = draw_bbox(overlay_panel, pred_bbox, color=(255, 50, 50), thickness=3)

    if gt_mask is not None:
        # GT mask outline (yellow) on right panel for comparison
        gt_contours, _ = cv2.findContours(
            gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay_panel, gt_contours, -1, (255, 220, 0), 2)

    # ── Legend labels (top-left) ─────────────────────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(orig_panel,    "Original + prompt bbox", (10, 30), font, 0.8, (60, 120, 255), 2)
    cv2.putText(overlay_panel, "Predicted mask (green)",  (10, 30), font, 0.8, (0, 230, 100), 2)
    cv2.putText(overlay_panel, "Predicted bbox (red)",    (10, 60), font, 0.8, (255, 50, 50), 2)

    if gt_mask is not None:
        cv2.putText(overlay_panel, "GT contour (yellow)", (10, 90), font, 0.8, (255, 220, 0), 2)

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union        = np.logical_or(pred_mask, gt_mask).sum()
        iou  = float(intersection) / float(union) if union > 0 else 0.0
        dice = (2.0 * float(intersection) / float(pred_mask.sum() + gt_mask.sum())
                if (pred_mask.sum() + gt_mask.sum()) > 0 else 0.0)
        hd95 = compute_hd95(pred_mask, gt_mask)

        hd95_str = f"{hd95:.2f}px" if np.isfinite(hd95) else "inf"
        metrics = {
            "Dice": f"{dice:.3f}",
            "IoU":  f"{iou:.3f}",
            "HD95": hd95_str,
        }
        overlay_panel = draw_metrics_topright(overlay_panel, metrics)
    else:
        cv2.putText(overlay_panel, "Metrics: N/A (no GT)", (10, 90), font, 0.8, (200, 200, 200), 2)
        iou  = None
        dice = None
        hd95 = None

    result_img = np.concatenate([orig_panel, overlay_panel], axis=1)
    return result_img, pred_mask, pred_bbox, gt_mask, iou, dice, hd95


def main():
    args = parse_args()

    # Resolve paths
    img_dir = os.path.abspath(args.img_dir)
    ckpt_path = os.path.abspath(args.checkpoint)
    result_dir = os.path.abspath(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)
    indiv_dir = os.path.join(result_dir, "individual")
    os.makedirs(indiv_dir, exist_ok=True)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # ── Load model ───────────────────────────────────────────────────────────
    print("Loading MedSAM2 model …")
    predictor = load_model(args.config, ckpt_path, device)
    print("Model loaded.")

    # ── Collect images ───────────────────────────────────────────────────────
    all_imgs = sorted(glob(os.path.join(img_dir, "*.jpg")) +
                      glob(os.path.join(img_dir, "*.jpeg")) +
                      glob(os.path.join(img_dir, "*.png")))

    if args.no_json_only:
        all_imgs = [
            p for p in all_imgs
            if not os.path.exists(os.path.splitext(p)[0] + ".json")
        ]
        print(f"Images without JSON annotation: {len(all_imgs)}")
    else:
        print(f"Total images found: {len(all_imgs)}")

    if args.max_images is not None:
        all_imgs = all_imgs[: args.max_images]
        print(f"Processing first {len(all_imgs)} images.")

    if not all_imgs:
        print("No images to process. Exiting.")
        return

    # ── Run inference ─────────────────────────────────────────────────────────
    grid_panels = []

    for img_path in tqdm(all_imgs, desc="Inference"):
        stem = Path(img_path).stem
        json_path = os.path.join(img_dir, stem + ".json")
        if not os.path.exists(json_path):
            json_path = None

        try:
            result_img, pred_mask, pred_bbox, gt_mask, iou, dice, hd95 = infer_image(
                predictor, img_path, json_path,
                bbox_pad=args.bbox_pad,
                device=device,
                prompt_mode=args.prompt_mode,
            )
            if iou is not None and dice is not None:
                hd95_str = f"{hd95:.2f}px" if (hd95 is not None and np.isfinite(hd95)) else "inf"
                tqdm.write(f"[{stem}] Dice: {dice:.4f} | IoU: {iou:.4f} | HD95: {hd95_str}")
            else:
                tqdm.write(f"[{stem}] no GT mask — metrics N/A")
        except Exception as e:
            print(f"\n[ERROR] Failed on {img_path}: {e}")
            continue

        # Save individual result image
        out_path = os.path.join(indiv_dir, stem + "_result.png")
        Image.fromarray(result_img).save(out_path)

        # Resize for the grid to keep memory manageable (max 512 px wide panel)
        max_panel_w = 512
        h, w = result_img.shape[:2]
        if w > max_panel_w * 2:
            scale = (max_panel_w * 2) / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            thumb = cv2.resize(result_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            thumb = result_img
        grid_panels.append(thumb)

    if not grid_panels:
        print("No successful predictions.")
        return

    print(f"\nBuilding grid ({len(grid_panels)} panels, {args.grid_cols} cols) …")
    grid = build_grid(grid_panels, ncols=args.grid_cols)
    grid_path = os.path.join(result_dir, "inference_grid.png")
    Image.fromarray(grid).save(grid_path)
    print(f"Grid saved: {grid_path}")
    print(f"Individual results saved: {indiv_dir}/")


if __name__ == "__main__":
    main()
