"""
Dense detector (kiểu FCOS) trên image-embedding ĐÔNG CỨNG của SAM2 (tiny@1024).
Lưới đặc trưng G×G (64×64, stride 16 trên ảnh 1024). Mỗi ô dự đoán:
  - objectness logit (ô có nằm trong khối u của bạn không)
  - box (cx,cy,w,h) chuẩn hoá [0,1] của ảnh gốc
Gán nhãn: ô DƯƠNG nếu tâm ô nằm trong box GT (gán box NHỎ NHẤT khi chồng); mỗi GT
luôn có ít nhất ô gần tâm nhất là dương. Suy luận: objectness>thr -> box -> NMS.

Ổn định + tiết kiệm mẫu hơn DETR-query (tránh "query collapse" trên dữ liệu nhỏ),
multi-box tự nhiên. Đúng yêu cầu: đề xuất box -> khớp box của bạn -> segment trong box.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- box ops (normalized cx,cy,w,h <-> xyxy) ----------------
def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def box_area(b):
    return (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)


def generalized_iou(a, b):
    """a:(N,4) b:(M,4) xyxy -> giou (N,M)."""
    area_a = box_area(a)[:, None]; area_b = box_area(b)[None, :]
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0); inter = wh[..., 0] * wh[..., 1]
    union = area_a + area_b - inter + 1e-7; iou = inter / union
    lt2 = torch.min(a[:, None, :2], b[None, :, :2])
    rb2 = torch.max(a[:, None, 2:], b[None, :, 2:])
    wh2 = (rb2 - lt2).clamp(min=0); enc = wh2[..., 0] * wh2[..., 1] + 1e-7
    return iou - (enc - union) / enc


def pairwise_iou(a, b):
    """a:(N,4) b:(M,4) xyxy -> iou (N,M)."""
    area_a = box_area(a)[:, None]; area_b = box_area(b)[None, :]
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0); inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a + area_b - inter + 1e-7)


# ---------------- dense detector head ----------------
class DenseDetector(nn.Module):
    def __init__(self, in_ch=256, hidden=256, grid=64, n_conv=4):
        super().__init__()
        self.grid = grid
        layers = []
        c = in_ch
        for _ in range(n_conv):
            layers += [nn.Conv2d(c, hidden, 3, padding=1), nn.GroupNorm(32, hidden), nn.ReLU()]
            c = hidden
        self.tower = nn.Sequential(*layers)
        self.obj = nn.Conv2d(hidden, 1, 3, padding=1)
        self.box = nn.Conv2d(hidden, 4, 3, padding=1)
        nn.init.constant_(self.obj.bias, -4.0)        # khởi tạo objectness thấp (ít dương)
        # lưới tâm ô chuẩn hoá [0,1] (đăng ký buffer để tự sang device)
        ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
        centers = torch.stack([(xs + 0.5) / grid, (ys + 0.5) / grid], -1).float()  # (G,G,2)
        self.register_buffer("centers", centers)

    def forward(self, feat):
        """feat:(B,C,Gf,Gf) -> obj_logits(B,G,G), boxes(B,G,G,4) cxcywh [0,1].
        Nếu lưới đích G khác độ phân giải feature (vd feat 64 nhưng grid=128 cho mảnh nhỏ)
        thì upsample feature lên (G,G) trước tower."""
        if feat.shape[-1] != self.grid:
            feat = F.interpolate(feat, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        x = self.tower(feat)
        obj = self.obj(x).squeeze(1)                          # (B,G,G)
        raw = self.box(x).permute(0, 2, 3, 1)                 # (B,G,G,4)
        # cx,cy = tâm ô + offset (tanh, phạm vi ±1 ô); w,h = sigmoid
        off = torch.tanh(raw[..., :2]) / self.grid
        ctr = self.centers[None] + off
        wh = raw[..., 2:].sigmoid()
        boxes = torch.cat([ctr, wh], -1).clamp(1e-4, 1 - 1e-4)
        return obj, boxes


# ---------------- gán nhãn dày đặc ----------------
@torch.no_grad()
def assign_targets(centers, gt_boxes, min_pos_per_gt=3):
    """centers:(G,G,2) tâm ô; gt_boxes:(M,4) cxcywh norm.
    -> pos_mask(G,G) bool, tgt_box(G,G,4), gi(G,G) chỉ số GT (cho mask-loss sau).
    min_pos_per_gt: ÉP top-k ô gần tâm nhất MỖI GT thành dương (center-sampling FCOS) —
    bảo đảm MẢNH U NHỎ (box < 1 ô lưới, không chứa tâm ô nào) vẫn có đủ ô dương để
    objectness học được (=1 trước đây → recall mảnh nhỏ kém)."""
    G = centers.shape[0]; dev = centers.device
    pos = torch.zeros(G, G, dtype=torch.bool, device=dev)
    tgt = torch.zeros(G, G, 4, device=dev)
    gidx = torch.full((G, G), -1, dtype=torch.long, device=dev)
    if gt_boxes.numel() == 0:
        return pos, tgt, gidx
    M = gt_boxes.shape[0]
    xy = centers.reshape(-1, 2)                               # (GG,2)
    gx = cxcywh_to_xyxy(gt_boxes)                             # (M,4)
    inside = ((xy[:, None, 0] >= gx[None, :, 0]) & (xy[:, None, 0] <= gx[None, :, 2]) &
              (xy[:, None, 1] >= gx[None, :, 1]) & (xy[:, None, 1] <= gx[None, :, 3]))  # (GG,M)
    areas = gt_boxes[:, 2] * gt_boxes[:, 3]                   # (M,)
    cost = torch.where(inside, areas[None].expand_as(inside), torch.full_like(inside, 1e9, dtype=torch.float))
    best_area, best_g = cost.min(1)                           # gán box NHỎ NHẤT chứa ô
    has = best_area < 1e8
    flat_pos = has.clone()
    flat_g = best_g.clone()
    # ÉP top-k ô gần tâm nhất MỖI GT thành dương (k=min_pos_per_gt). GT nhỏ -> đủ tín hiệu.
    d = torch.cdist(gt_boxes[:, :2], xy)                      # (M,GG)
    k = min(max(1, min_pos_per_gt), xy.shape[0])
    near_k = d.topk(k, dim=1, largest=False).indices.reshape(-1)        # (M*k,)
    gt_ids = torch.arange(M, device=dev).repeat_interleave(k)          # (M*k,)
    flat_pos[near_k] = True
    flat_g[near_k] = gt_ids                                            # ô ép-dương -> GT của nó
    pos = flat_pos.reshape(G, G)
    gidx = flat_g.reshape(G, G)
    gidx[~pos] = -1
    tgt = gt_boxes[flat_g.clamp(min=0)].reshape(G, G, 4)
    return pos, tgt, gidx


def dense_losses(obj_logits, boxes, pos, tgt_box, focal_alpha=0.25, focal_gamma=2.0):
    """1 ảnh: obj_logits(G,G), boxes(G,G,4), pos(G,G), tgt_box(G,G,4)."""
    tgt_obj = pos.float()
    p = obj_logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(obj_logits, tgt_obj, reduction="none")
    pt = p * tgt_obj + (1 - p) * (1 - tgt_obj)
    alpha_t = focal_alpha * tgt_obj + (1 - focal_alpha) * (1 - tgt_obj)
    focal = alpha_t * (1 - pt) ** focal_gamma * ce
    npos = max(1, int(pos.sum()))
    cls = focal.sum() / npos
    if pos.any():
        pb = boxes[pos]; gb = tgt_box[pos]
        l1 = F.l1_loss(pb, gb, reduction="mean")
        giou = (1 - torch.diag(generalized_iou(cxcywh_to_xyxy(pb), cxcywh_to_xyxy(gb)))).mean()
    else:
        z = boxes.sum() * 0.0; l1 = z; giou = z
    return {"cls": cls, "l1": l1, "giou": giou}


@torch.no_grad()
def decode_detections(obj_logits, boxes, thr=0.3, nms_iou=0.5, max_box=20):
    """1 ảnh: obj_logits(G,G), boxes(G,G,4) -> (boxes_kept cxcywh (n,4), scores (n,))."""
    prob = obj_logits.sigmoid().reshape(-1)
    bx = boxes.reshape(-1, 4)
    keep = prob > thr
    if keep.sum() == 0:
        keep = torch.zeros_like(prob, dtype=torch.bool); keep[prob.argmax()] = True
    p = prob[keep]; b = bx[keep]
    # NMS
    order = p.argsort(descending=True)
    xy = cxcywh_to_xyxy(b)
    kept = []
    while order.numel() > 0 and len(kept) < max_box:
        i = order[0].item(); kept.append(i)
        if order.numel() == 1: break
        ious = pairwise_iou(xy[i:i+1], xy[order[1:]])[0]
        order = order[1:][ious <= nms_iou]
    idx = torch.tensor(kept, device=obj_logits.device, dtype=torch.long)
    return b[idx], p[idx]


def _decode_px(obj_logits, boxes, H, W, thr):
    """decode -> box pixel xyxy (n,4) float32 (numpy)."""
    import numpy as np
    bxk, _ = decode_detections(obj_logits, boxes, thr=thr)
    xyxy = cxcywh_to_xyxy(bxk).clamp(0, 1).cpu().numpy()
    return (xyxy * np.array([W, H, W, H], np.float32)).astype(np.float32)


def gate_by_specimen(boxes_px, spec):
    """Bỏ box có TÂM nằm ngoài bệnh phẩm (= thước/nhãn/nền). spec rỗng -> giữ hết.
    -> (boxes_kept, dropped_mask bool)."""
    import numpy as np
    if spec is None or spec.sum() == 0 or len(boxes_px) == 0:
        return boxes_px, np.zeros(len(boxes_px), bool)
    H, W = spec.shape
    keep = np.ones(len(boxes_px), bool)
    for i, b in enumerate(boxes_px):
        cx = int(np.clip((b[0] + b[2]) / 2, 0, W - 1)); cy = int(np.clip((b[1] + b[3]) / 2, 0, H - 1))
        keep[i] = spec[cy, cx] > 0
    return boxes_px[keep], ~keep


@torch.no_grad()
def propose_boxes(obj_logits, boxes, H, W, spec=None, thr=0.5,
                  fallback_thr=0.35, fallback_if_le=1):
    """Đường INFERENCE chính thức (validate trên test vẽ tay):
      1) decode objectness>thr -> box pixel xyxy
      2) GATE specimen: bỏ box rơi vào thước/nhãn/nền (clean_specimen mask)
      3) FALLBACK recall đa-u: nếu sau gate còn <=fallback_if_le box, giải lại ở
         fallback_thr (thấp hơn) + gate lại -> cứu ca detector quá dè dặt (1box/nhiều-u).
    obj_logits(G,G), boxes(G,G,4) là output detector cho 1 ảnh (forward 1 lần, decode rẻ).
    -> boxes_px (n,4) xyxy float32."""
    px = _decode_px(obj_logits, boxes, H, W, thr)
    px, _ = gate_by_specimen(px, spec)
    if len(px) <= fallback_if_le and fallback_thr is not None and fallback_thr < thr:
        px2 = _decode_px(obj_logits, boxes, H, W, fallback_thr)
        px2, _ = gate_by_specimen(px2, spec)
        if len(px2) > len(px):
            px = px2
    return px
