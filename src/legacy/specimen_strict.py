"""Specimen-aware detector decoding utilities.

The old inference path only checked whether a decoded box center was inside the
specimen. This module adds stricter constraints:
  - suppress detector cells whose grid center is outside the specimen;
  - reject boxes that mostly cover background;
  - optionally shrink each box to the specimen pixels inside it;
  - always clip the final predicted mask to the specimen mask.
"""
import cv2
import numpy as np
import torch

from detector import cxcywh_to_xyxy, pairwise_iou


@torch.no_grad()
def decode_detections_specimen(
    obj_logits,
    boxes,
    h,
    w,
    spec,
    thr=0.5,
    nms_iou=0.5,
    max_box=20,
    min_box_spec_frac=0.55,
    shrink_to_spec=True,
    fallback_thr=None,
):
    """Decode dense detector outputs with specimen constraints.

    Returns boxes in pixel xyxy float32. `spec` is uint8/bool HxW specimen mask.
    """
    px = _decode_once(obj_logits, boxes, h, w, spec, thr, nms_iou, max_box)
    px = _filter_and_shrink(px, spec, min_box_spec_frac, shrink_to_spec)
    if len(px) == 0 and fallback_thr is not None and fallback_thr < thr:
        px = _decode_once(obj_logits, boxes, h, w, spec, fallback_thr, nms_iou, max_box)
        px = _filter_and_shrink(px, spec, min_box_spec_frac, shrink_to_spec)
    return px


@torch.no_grad()
def _decode_once(obj_logits, boxes, h, w, spec, thr, nms_iou, max_box):
    prob = obj_logits.sigmoid().reshape(-1)
    bx = boxes.reshape(-1, 4)
    if spec is not None and spec.sum() > 0:
        g = obj_logits.shape[-1]
        spec_grid = _spec_grid(spec, g).reshape(-1).to(prob.device)
        prob = prob.masked_fill(~spec_grid, 0.0)
    keep = prob > thr
    if keep.sum() == 0:
        if prob.max() <= 0:
            return np.zeros((0, 4), np.float32)
        keep = torch.zeros_like(prob, dtype=torch.bool)
        keep[prob.argmax()] = True
    p = prob[keep]
    b = bx[keep]
    order = p.argsort(descending=True)
    xy = cxcywh_to_xyxy(b).clamp(0, 1)
    kept = []
    while order.numel() > 0 and len(kept) < max_box:
        i = order[0].item()
        kept.append(i)
        if order.numel() == 1:
            break
        ious = pairwise_iou(xy[i : i + 1], xy[order[1:]])[0]
        order = order[1:][ious <= nms_iou]
    if not kept:
        return np.zeros((0, 4), np.float32)
    idx = torch.tensor(kept, device=obj_logits.device, dtype=torch.long)
    xyxy = xy[idx].cpu().numpy()
    return (xyxy * np.array([w, h, w, h], np.float32)).astype(np.float32)


def _spec_grid(spec, g):
    spec_u8 = (spec > 0).astype(np.uint8)
    small = cv2.resize(spec_u8, (g, g), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(small > 0.10)


def _filter_and_shrink(boxes_px, spec, min_box_spec_frac, shrink_to_spec):
    if spec is None or spec.sum() == 0 or len(boxes_px) == 0:
        return boxes_px.astype(np.float32)
    h, w = spec.shape
    out = []
    sp = spec > 0
    for b in boxes_px:
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))
        reg = sp[y0:y1, x0:x1]
        frac = float(reg.sum()) / max(1, reg.size)
        if frac < min_box_spec_frac:
            continue
        if shrink_to_spec:
            ys, xs = np.where(reg)
            if len(xs) == 0:
                continue
            x0 = x0 + int(xs.min())
            x1 = x0 + int(xs.max() - xs.min() + 1)
            y0 = y0 + int(ys.min())
            y1 = y0 + int(ys.max() - ys.min() + 1)
        out.append([x0, y0, x1, y1])
    return np.asarray(out, np.float32)


def specimen_post_mask(mask, spec, min_frac=0.001):
    """Clip to specimen and remove tiny fragments. Keeps multiple components."""
    if spec is not None and spec.sum() > 0:
        mask = mask & (spec > 0)
    if mask.sum() == 0:
        return mask
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask, bool)
    thr = min_frac * mask.size
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= thr:
            out |= lab == i
    return out
