"""
LOCALIZE #2 — BOX REFINER: detector cho box "hòm hòm" (IoU~0.55 vs GT-mảnh) -> CNN hồi quy
siết về box khít. Self-supervised từ mask THẬT: mô phỏng box lỏng (jitter GT-mảnh xuống
IoU~0.5-0.7) -> học hồi quy về box GT. Inference: áp lên box detector trước khi segment.
  python box_refiner.py --epochs 80
Lưu checkpoints/box_refiner.pt. Env: medsam2_anno.
"""
import argparse, csv, json, os, sys, random, hashlib
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from seg_crop import frag_boxes, set_seed, TRUTH_DIR, pat
DEVICE = "cuda"; CTX = 0.35; RSIZE = 256; IMG_DIR = "data/20241212"
CKPT = "checkpoints/box_refiner.pt"

class RefinerNet(nn.Module):
    """crop (3ch) -> 4 toạ độ box khít (normalized [0,1] trong crop)."""
    def __init__(self):
        super().__init__()
        def cb(i, o, s): return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.GELU())
        self.net = nn.Sequential(cb(3, 32, 2), cb(32, 64, 2), cb(64, 128, 2), cb(128, 128, 2),
                                 cb(128, 256, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 4))
    def forward(self, x): return torch.sigmoid(self.head(self.net(x)))  # x0,y0,x1,y1 in [0,1]

def ctx_crop(bgr, box, ctx, rsize):
    """crop quanh box + ctx pad -> (crop resize, (cx0,cy0,cw,ch) vùng crop trên ảnh gốc)."""
    H, W = bgr.shape[:2]; x0, y0, x1, y1 = box; bw = x1 - x0; bh = y1 - y0
    cx0 = max(0, int(x0 - bw * ctx)); cy0 = max(0, int(y0 - bh * ctx))
    cx1 = min(W, int(x1 + bw * ctx)); cy1 = min(H, int(y1 + bh * ctx))
    if cx1 <= cx0: cx1 = min(W, cx0 + 1)
    if cy1 <= cy0: cy1 = min(H, cy0 + 1)
    crop = cv2.resize(bgr[cy0:cy1, cx0:cx1], (rsize, rsize), interpolation=cv2.INTER_AREA)
    return crop, (cx0, cy0, cx1 - cx0, cy1 - cy0)

def jitter_to_loose(gt, W, H):
    """jitter box GT-mảnh -> box lỏng (mô phỏng detector, IoU~0.5-0.7)."""
    x0, y0, x1, y1 = gt; bw = x1 - x0; bh = y1 - y0
    sx = random.uniform(-0.25, 0.25) * bw; sy = random.uniform(-0.25, 0.25) * bh
    ex = random.uniform(-0.15, 0.35); ey = random.uniform(-0.15, 0.35)
    nx0 = x0 + sx - bw * ex / 2; nx1 = x1 + sx + bw * ex / 2
    ny0 = y0 + sy - bh * ey / 2; ny1 = y1 + sy + bh * ey / 2
    return [max(0, nx0), max(0, ny0), min(W, nx1), min(H, ny1)]

class RefData(torch.utils.data.Dataset):
    def __init__(self, items, train=True): self.items = items; self.train = train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        stem, gt = self.items[i]
        bgr = cv2.imread(f"{IMG_DIR}/{stem}.jpg"); H, W = bgr.shape[:2]
        loose = jitter_to_loose(gt, W, H) if self.train else gt
        crop, (cx0, cy0, cw, ch) = ctx_crop(bgr, loose, CTX, RSIZE)
        # target = GT box trong khung crop, normalized
        tx0 = (gt[0] - cx0) / cw; ty0 = (gt[1] - cy0) / ch
        tx1 = (gt[2] - cx0) / cw; ty1 = (gt[3] - cy0) / ch
        t = np.clip([tx0, ty0, tx1, ty1], 0, 1).astype(np.float32)
        if self.train and random.random() < 0.5:
            crop = crop[:, ::-1]; t = np.array([1 - t[2], t[1], 1 - t[0], t[3]], np.float32)
        rgb = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(t)

def giou_loss(p, t):
    """p,t: (B,4) x0y0x1y1 [0,1]."""
    px0, py0, px1, py1 = p.unbind(-1); tx0, ty0, tx1, ty1 = t.unbind(-1)
    px1 = torch.maximum(px1, px0 + 1e-3); py1 = torch.maximum(py1, py0 + 1e-3)
    ix0 = torch.maximum(px0, tx0); iy0 = torch.maximum(py0, ty0)
    ix1 = torch.minimum(px1, tx1); iy1 = torch.minimum(py1, ty1)
    iw = (ix1 - ix0).clamp(min=0); ih = (iy1 - iy0).clamp(min=0); inter = iw * ih
    ap = (px1 - px0) * (py1 - py0); at = (tx1 - tx0) * (ty1 - ty0); union = ap + at - inter + 1e-6
    iou = inter / union
    ex0 = torch.minimum(px0, tx0); ey0 = torch.minimum(py0, ty0)
    ex1 = torch.maximum(px1, tx1); ey1 = torch.maximum(py1, ty1)
    area_c = (ex1 - ex0) * (ey1 - ey0) + 1e-6
    giou = iou - (area_c - union) / area_c
    return (1 - giou).mean()

@torch.no_grad()
def refine_box(net, bgr, box):
    """áp refiner: box lỏng -> box khít trên ảnh gốc."""
    crop, (cx0, cy0, cw, ch) = ctx_crop(bgr, list(map(float, box)), CTX, RSIZE)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(np.ascontiguousarray(((rgb - 0.5) / 0.5).transpose(2, 0, 1)))[None].to(DEVICE)
    p = net(x)[0].cpu().numpy()
    return [cx0 + p[0] * cw, cy0 + p[1] * ch, cx0 + p[2] * cw, cy0 + p[3] * ch]

def load_refiner():
    net = RefinerNet().to(DEVICE); net.load_state_dict(torch.load(CKPT, weights_only=False)["net"]); net.eval()
    return net

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args(); set_seed(0)
    truth = sorted(f[:-4] for f in os.listdir(TRUTH_DIR) if f.endswith(".png"))
    hd = set(map(pat, json.load(open("labels_handdraw/select.json"))["stems"]))
    e200 = set(pat(r["stem"]) for r in csv.DictReader(open("results/confirm200_ft_vs_zs.csv")))
    t12 = set(map(pat, json.load(open("labels/test_frozen.json"))["test"]))
    excl = hd | e200 | t12
    clean = [s for s in truth if pat(s) not in excl]
    clean.sort(key=lambda s: hashlib.md5(s.encode()).hexdigest())
    val_stems = set(clean[:14])
    items_tr, items_va = [], []
    for s in clean:
        m = cv2.imread(f"{TRUTH_DIR}/{s}.png", 0) > 127
        for b in frag_boxes(m):
            (items_va if s in val_stems else items_tr).append((s, b))
    print(f"[refiner] train={len(items_tr)} val={len(items_va)} frag", flush=True)
    tr = torch.utils.data.DataLoader(RefData(items_tr, True), batch_size=args.batch, shuffle=True,
                                     num_workers=4, drop_last=True)
    va = torch.utils.data.DataLoader(RefData(items_va, False), batch_size=args.batch, num_workers=2)
    net = RefinerNet().to(DEVICE); opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    def viou(p, t):
        return 1 - giou_loss(p, t).item()  # xấp xỉ (mean giou)
    best = -1
    for ep in range(1, args.epochs + 1):
        net.train(); tot = 0; nb = 0
        for x, t in tr:
            x = x.to(DEVICE); t = t.to(DEVICE); p = net(x)
            loss = F.l1_loss(p, t) + giou_loss(p, t)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        sched.step()
        net.eval(); gv = []
        with torch.no_grad():
            for x, t in va:
                p = net(x.to(DEVICE)); gv.append(1 - giou_loss(p, t.to(DEVICE)).item())
        vg = float(np.mean(gv)) if gv else 0; star = ""
        if vg > best: best = vg; torch.save({"net": net.state_dict(), "giou": vg, "epoch": ep}, CKPT); star = "  *best"
        if ep % 5 == 0 or star: print(f"[ep {ep}] loss={tot/max(nb,1):.4f} val_giou≈{vg:.4f}{star}", flush=True)
    print(f"\nRefiner xong. best val giou≈{best:.4f} -> {CKPT}", flush=True)

if __name__ == "__main__":
    main()
