"""Component-aware specimen-strict inference utilities.

This keeps the good property of specimen-strict inference (no background spill),
but treats each disconnected specimen slice as its own decoding region. This is
intended for images that contain several cut slices: a high-confidence slice
should not consume the whole image-level box budget while other slices get no
box.
"""
import cv2
import numpy as np
import torch

from detector import cxcywh_to_xyxy, pairwise_iou


def specimen_components(spec, min_area_frac=0.004, max_components=12):
    """Return large connected specimen components as uint8 masks.

    The components are sorted top-to-bottom then left-to-right to keep output
    stable. Very small tissue fragments are ignored because they mostly create
    false positives in fat scraps or torn edges.
    """
    if spec is None or spec.sum() == 0:
        return []
    sp = (spec > 0).astype(np.uint8)
    n, lab, st, cent = cv2.connectedComponentsWithStats(sp, 8)
    h, w = sp.shape
    min_area = int(round(min_area_frac * h * w))
    comps = []
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(st[i, cv2.CC_STAT_LEFT])
        y = int(st[i, cv2.CC_STAT_TOP])
        ww = int(st[i, cv2.CC_STAT_WIDTH])
        hh = int(st[i, cv2.CC_STAT_HEIGHT])
        cx, cy = cent[i]
        comps.append({
            "id": i,
            "area": area,
            "bbox": (x, y, x + ww, y + hh),
            "center": (float(cx), float(cy)),
            "mask": (lab == i).astype(np.uint8),
        })
    comps.sort(key=lambda c: (c["center"][1], c["center"][0]))
    return comps[:max_components]


def decode_detections_components(
    obj_logits,
    boxes,
    h,
    w,
    spec,
    thr=0.5,
    nms_iou=0.5,
    max_box_per_component=4,
    min_box_spec_frac=0.45,
    shrink_to_spec=True,
    fallback_thr=0.25,
    min_component_area_frac=0.004,
    max_components=12,
):
    """Decode boxes per disconnected specimen slice.

    Returns:
      boxes_px: Nx4 xyxy float32
      comp_ids: N int component indices into returned components
      comps: component metadata from specimen_components()
    """
    comps = specimen_components(spec, min_component_area_frac, max_components)
    if not comps:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int32), []

    all_boxes = []
    all_comp_ids = []
    for ci, comp in enumerate(comps):
        bx = _decode_component_thresholded(
            obj_logits,
            boxes,
            h,
            w,
            comp["mask"],
            thr=thr,
            nms_iou=nms_iou,
            max_box=max_box_per_component,
            min_box_spec_frac=min_box_spec_frac,
            shrink_to_spec=shrink_to_spec,
        )
        if len(bx) == 0 and fallback_thr is not None and fallback_thr < thr:
            bx = _decode_component_thresholded(
                obj_logits,
                boxes,
                h,
                w,
                comp["mask"],
                thr=fallback_thr,
                nms_iou=nms_iou,
                max_box=max_box_per_component,
                min_box_spec_frac=min_box_spec_frac,
                shrink_to_spec=shrink_to_spec,
            )
        for b in bx:
            all_boxes.append(b)
            all_comp_ids.append(ci)

    if not all_boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int32), comps
    return np.asarray(all_boxes, np.float32), np.asarray(all_comp_ids, np.int32), comps


@torch.no_grad()
def _decode_component_thresholded(
    obj_logits,
    boxes,
    h,
    w,
    comp_mask,
    thr,
    nms_iou,
    max_box,
    min_box_spec_frac,
    shrink_to_spec,
):
    """Decode inside one component without forcing an argmax fallback."""
    prob = obj_logits.sigmoid().reshape(-1)
    bx = boxes.reshape(-1, 4)
    g = obj_logits.shape[-1]
    spec_grid = _spec_grid(comp_mask, g).reshape(-1).to(prob.device)
    prob = prob.masked_fill(~spec_grid, 0.0)
    keep = prob > thr
    if keep.sum() == 0:
        return np.zeros((0, 4), np.float32)

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
    px = (xyxy * np.array([w, h, w, h], np.float32)).astype(np.float32)
    return _filter_and_shrink(px, comp_mask, min_box_spec_frac, shrink_to_spec)


def _spec_grid(spec, g):
    spec_u8 = (spec > 0).astype(np.uint8)
    small = cv2.resize(spec_u8, (g, g), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(small > 0.10)


def _filter_and_shrink(boxes_px, comp_mask, min_box_spec_frac, shrink_to_spec):
    if comp_mask is None or comp_mask.sum() == 0 or len(boxes_px) == 0:
        return boxes_px.astype(np.float32)
    h, w = comp_mask.shape
    sp = comp_mask > 0
    out = []
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


def clip_mask_to_component(mask, comp_mask):
    """Clip SAM output to one specimen component and remove tiny fragments."""
    out = mask.astype(bool) & (comp_mask > 0)
    if out.sum() == 0:
        return out
    n, lab, st, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), 8)
    keep = np.zeros_like(out, bool)
    min_area = max(32, int(round(0.001 * comp_mask.size)))
    for i in range(1, n):
        if int(st[i, cv2.CC_STAT_AREA]) >= min_area:
            keep |= lab == i
    return keep
