"""
LOCALIZE #5 — SLIC superpixel + classifier. Băm ảnh thành superpixel (mảnh cùng màu/vân),
phân loại từng mảnh U/mô-lành (MLP trên đặc trưng màu+texture+vị trí), hợp mảnh-U -> box khít
theo ranh giới vân tự nhiên. Train trên 178 mask thật. Lưu checkpoints/slic_clf.pt.
  python slic_clf.py --epochs 60
Env: medsam2_anno.
"""
import argparse, csv, json, os, sys, hashlib
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from skimage.segmentation import slic
from seg_crop import set_seed, TRUTH_DIR, pat, frag_boxes
from seg_crop import texture_chan
from specimen_clean import clean_specimen
DEVICE = "cuda"; IMG_DIR = "data/20241212"; CKPT = "checkpoints/slic_clf.pt"
N_SEG = 800; FEATD = 14

def sp_features(bgr, spec):
    """SLIC trên ảnh -> (labels, feats[K,FEATD], centers, valid_mask). Chỉ mảnh trong specimen."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    tex = texture_chan(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    H, W = bgr.shape[:2]
    labels = slic(rgb, n_segments=N_SEG, compactness=12, start_label=0, channel_axis=-1)
    K = labels.max() + 1
    feats = np.zeros((K, FEATD), np.float32); cnt = np.zeros(K, np.float32)
    ys, xs = np.mgrid[0:H, 0:W]
    flat = labels.ravel()
    def accum(ch, idx):
        np.add.at(feats[:, idx], flat, ch.ravel())
    for c in range(3): accum(rgb[..., c] / 255.0, c)
    for c in range(3): accum(lab[..., c] / 255.0, 3 + c)
    accum(tex, 6)
    accum(xs / W, 7); accum(ys / H, 8)
    np.add.at(cnt, flat, 1.0)
    cnt = np.maximum(cnt, 1)
    for i in range(9): feats[:, i] /= cnt
    # std màu (sần) cho 3 kênh rgb
    for c in range(3):
        sq = np.zeros(K, np.float32); np.add.at(sq, flat, (rgb[..., c].ravel() / 255.0) ** 2)
        feats[:, 9 + c] = np.sqrt(np.maximum(sq / cnt - feats[:, c] ** 2, 0))
    feats[:, 12] = cnt / (H * W) * 1000  # diện tích tương đối
    # tỉ lệ pixel trong specimen
    inq = np.zeros(K, np.float32); np.add.at(inq, flat, (spec.ravel() > 0).astype(np.float32))
    feats[:, 13] = inq / cnt
    return labels, feats, K

class MLP(nn.Module):
    def __init__(self, d=FEATD):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x)

def load_clf():
    d = torch.load(CKPT, weights_only=False); net = MLP().to(DEVICE); net.load_state_dict(d["net"]); net.eval()
    return net, d["mean"], d["std"], d["thr"]

@torch.no_grad()
def slic_boxes(bgr, spec, net, mean, std):
    labels, feats, K = sp_features(bgr, spec)
    x = torch.from_numpy((feats - mean) / std).float().to(DEVICE)
    prob = torch.sigmoid(net(x))[:, 0].cpu().numpy()
    d = torch.load(CKPT, weights_only=False); thr = d["thr"]
    tum = np.zeros(labels.shape, np.uint8)
    pos = set(np.where((prob > thr) & (feats[:, 13] > 0.5))[0].tolist())
    mask = np.isin(labels, list(pos)) if pos else np.zeros(labels.shape, bool)
    tum[mask] = 1
    tum = cv2.morphologyEx(tum, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(tum, 8); boxes = []
    thr_a = 0.002 * tum.size
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < thr_a: continue
        x0, y0 = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
        boxes.append([float(x0), float(y0), float(x0 + st[i, cv2.CC_STAT_WIDTH]), float(y0 + st[i, cv2.CC_STAT_HEIGHT])])
    return boxes

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=60); args = ap.parse_args(); set_seed(0)
    truth = sorted(f[:-4] for f in os.listdir(TRUTH_DIR) if f.endswith(".png"))
    hd = set(map(pat, json.load(open("labels_handdraw/select.json"))["stems"]))
    e200 = set(pat(r["stem"]) for r in csv.DictReader(open("results/confirm200_ft_vs_zs.csv")))
    t12 = set(map(pat, json.load(open("labels/test_frozen.json"))["test"]))
    clean = [s for s in truth if pat(s) not in (hd | e200 | t12)]
    print(f"[slic] build features từ {len(clean)} ảnh...", flush=True)
    X, Y = [], []
    for s in clean:
        bgr = cv2.imread(f"{IMG_DIR}/{s}.jpg"); m = cv2.imread(f"{TRUTH_DIR}/{s}.png", 0) > 127
        spec, _, _ = clean_specimen(bgr)
        labels, feats, K = sp_features(bgr, spec)
        flat = labels.ravel(); tum = np.zeros(K); cnt = np.zeros(K)
        np.add.at(tum, flat, m.ravel().astype(np.float32)); np.add.at(cnt, flat, 1.0)
        frac = tum / np.maximum(cnt, 1)
        X.append(feats); Y.append((frac > 0.5).astype(np.float32))
    X = np.concatenate(X); Y = np.concatenate(Y)
    mean = X.mean(0); std = X.std(0) + 1e-6
    print(f"[slic] superpixels={len(X)} pos={int(Y.sum())} ({100*Y.mean():.1f}%)", flush=True)
    Xt = torch.from_numpy((X - mean) / std).float().to(DEVICE); Yt = torch.from_numpy(Y).float().to(DEVICE)[:, None]
    net = MLP().to(DEVICE); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    posw = torch.tensor([(Y == 0).sum() / max(1, (Y == 1).sum())]).to(DEVICE)
    for ep in range(1, args.epochs + 1):
        net.train(); idx = torch.randperm(len(Xt))
        for i in range(0, len(idx), 4096):
            b = idx[i:i + 4096]; lg = net(Xt[b])
            loss = F.binary_cross_entropy_with_logits(lg, Yt[b], pos_weight=posw)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0: print(f"[ep {ep}] loss={float(loss):.4f}", flush=True)
    # chọn thr theo F1 train
    net.eval()
    with torch.no_grad(): pr = torch.sigmoid(net(Xt))[:, 0].cpu().numpy()
    best_thr, best_f1 = 0.5, -1
    for thr in np.arange(0.3, 0.9, 0.05):
        pp = pr > thr; tp = (pp & (Y > 0.5)).sum(); fp = (pp & (Y < 0.5)).sum(); fn = ((~pp) & (Y > 0.5)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
        if f1 > best_f1: best_f1, best_thr = f1, float(thr)
    torch.save({"net": net.state_dict(), "mean": mean, "std": std, "thr": best_thr}, CKPT)
    print(f"[slic] thr={best_thr:.2f} F1={best_f1:.3f} -> {CKPT}", flush=True)

if __name__ == "__main__":
    main()
