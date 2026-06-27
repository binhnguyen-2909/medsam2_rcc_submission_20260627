"""
LOCALIZE #6 — Grid patch classifier -> heatmap -> box. CNN nhẹ phân loại nhị phân patch
chứa-U/không -> quét toàn ảnh thành heatmap xác suất -> threshold -> box định vị.
Train trên 178 mask thật. Lưu checkpoints/grid_clf.pt.
  python grid_clf.py --epochs 30
Env: medsam2_anno.
"""
import argparse, csv, json, os, sys, random, hashlib
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from seg_crop import set_seed, TRUTH_DIR, pat
from specimen_clean import clean_specimen
DEVICE = "cuda"; IMG_DIR = "data/20241212"; CKPT = "checkpoints/grid_clf.pt"
PATCH = 160; PSIZE = 96

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        def cb(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.GroupNorm(8, o), nn.GELU())
        self.f = nn.Sequential(cb(3, 32), cb(32, 64), cb(64, 128), cb(128, 128),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.h = nn.Linear(128, 1)
    def forward(self, x): return self.h(self.f(x))

def patch_tensor(crop):
    cr = cv2.resize(crop, (PSIZE, PSIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(cr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(((rgb - 0.5) / 0.5).transpose(2, 0, 1)))

def sample_patches(bgr, m, spec, train=True):
    """trả (patches tensor, labels) — tumor nếu center 1/2 patch phủ GT>10%."""
    H, W = bgr.shape[:2]; xs, ys, labs = [], [], []
    step = PATCH // 2 if train else PATCH // 2
    coords = []
    for y in range(0, H - PATCH + 1, step):
        for x in range(0, W - PATCH + 1, step):
            cx, cy = x + PATCH // 2, y + PATCH // 2
            if spec[cy, cx] == 0: continue
            coords.append((x, y))
    pts, lbs = [], []
    for x, y in coords:
        c = PATCH // 4
        center = m[y + c:y + 3 * c, x + c:x + 3 * c]
        lab = 1.0 if center.mean() > 0.10 else 0.0
        pts.append(patch_tensor(bgr[y:y + PATCH, x:x + PATCH])); lbs.append(lab)
    return pts, lbs

def load_grid():
    net = TinyNet().to(DEVICE); net.load_state_dict(torch.load(CKPT, weights_only=False)["net"]); net.eval()
    return net

@torch.no_grad()
def grid_boxes(bgr, spec, net, thr=0.5):
    H, W = bgr.shape[:2]; step = PATCH // 3
    heat = np.zeros((H, W), np.float32); wts = np.zeros((H, W), np.float32)
    batch = []; locs = []
    for y in range(0, H - PATCH + 1, step):
        for x in range(0, W - PATCH + 1, step):
            cx, cy = x + PATCH // 2, y + PATCH // 2
            if spec[cy, cx] == 0: continue
            batch.append(patch_tensor(bgr[y:y + PATCH, x:x + PATCH])); locs.append((x, y))
    for i in range(0, len(batch), 256):
        xb = torch.stack(batch[i:i + 256]).to(DEVICE); pr = torch.sigmoid(net(xb))[:, 0].cpu().numpy()
        for j, (x, y) in enumerate(locs[i:i + 256]):
            heat[y:y + PATCH, x:x + PATCH] += pr[j]; wts[y:y + PATCH, x:x + PATCH] += 1
    heat = heat / np.maximum(wts, 1e-6)
    binm = (heat > thr).astype(np.uint8)
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(binm, 8); boxes = []
    thr_a = 0.002 * binm.size
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < thr_a: continue
        x0, y0 = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
        boxes.append([float(x0), float(y0), float(x0 + st[i, cv2.CC_STAT_WIDTH]), float(y0 + st[i, cv2.CC_STAT_HEIGHT])])
    return boxes

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=30); args = ap.parse_args(); set_seed(0)
    truth = sorted(f[:-4] for f in os.listdir(TRUTH_DIR) if f.endswith(".png"))
    hd = set(map(pat, json.load(open("labels_handdraw/select.json"))["stems"]))
    e200 = set(pat(r["stem"]) for r in csv.DictReader(open("results/confirm200_ft_vs_zs.csv")))
    t12 = set(map(pat, json.load(open("labels/test_frozen.json"))["test"]))
    clean = [s for s in truth if pat(s) not in (hd | e200 | t12)]
    print(f"[grid] trích patch từ {len(clean)} ảnh...", flush=True)
    P, L = [], []
    for s in clean:
        bgr = cv2.imread(f"{IMG_DIR}/{s}.jpg"); m = cv2.imread(f"{TRUTH_DIR}/{s}.png", 0) > 127
        spec, _, _ = clean_specimen(bgr)
        pts, lbs = sample_patches(bgr, m.astype(np.float32), spec, True)
        P += pts; L += lbs
    L = np.array(L, np.float32)
    print(f"[grid] patches={len(P)} pos={int(L.sum())} ({100*L.mean():.1f}%)", flush=True)
    net = TinyNet().to(DEVICE); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    posw = torch.tensor([(L == 0).sum() / max(1, (L == 1).sum())]).to(DEVICE)
    Lt = torch.from_numpy(L).to(DEVICE)[:, None]
    for ep in range(1, args.epochs + 1):
        net.train(); idx = torch.randperm(len(P)); tot = 0; nb = 0
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]; xb = torch.stack([P[j] for j in b]).to(DEVICE)
            lg = net(xb); loss = F.binary_cross_entropy_with_logits(lg, Lt[b], pos_weight=posw)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        if ep % 5 == 0: print(f"[ep {ep}] loss={tot/max(nb,1):.4f}", flush=True)
    torch.save({"net": net.state_dict()}, CKPT)
    print(f"[grid] -> {CKPT}", flush=True)

if __name__ == "__main__":
    main()
