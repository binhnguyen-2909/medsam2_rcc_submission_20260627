"""
#G DENSE CONTRASTIVE (SupCon/InfoNCE) — đánh GỐC RỄ: ép vector vân-U vs vân-LÀNH xa nhau
trong latent space. Patch chứa u = positive, thận lành = negative; SupCon loss thay BCE/Dice.
ĐO TRỰC TIẾP BỨC TƯỜNG: AUC phân biệt u/lành của feature contrastive (cao = tách được).
Rồi localizer: encoder+linear -> heatmap -> box -> segmenter -> Dice 50 vẽ tay.
  python supcon_loc.py --epochs 40
Env: medsam2_anno.
"""
import argparse, csv, json, os, sys, random, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from seg_crop import set_seed, TRUTH_DIR, pat
from specimen_clean import clean_specimen
DEVICE = "cuda"; IMG_DIR = "data/20241212"; CKPT = "checkpoints/supcon.pt"; PSZ = 96

class Enc(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        def cb(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.GroupNorm(8, o), nn.GELU())
        self.f = nn.Sequential(cb(3, 32), cb(32, 64), cb(64, 128), cb(128, 128),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, dim))
        self.cls = nn.Linear(128, 1)
    def feat(self, x): return self.f(x)
    def forward(self, x):
        h = self.f(x); return F.normalize(self.proj(h), dim=1), self.cls(h)

def supcon(z, y, t=0.1):
    """SupCon loss. z (N,d) chuẩn hóa, y (N,) nhãn 0/1."""
    sim = z @ z.t() / t; N = z.shape[0]
    mask = (y[:, None] == y[None, :]).float(); eye = torch.eye(N, device=z.device)
    mask = mask - eye
    logits = sim - eye * 1e9
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    denom = mask.sum(1).clamp(min=1)
    return -((mask * logp).sum(1) / denom).mean()

def patch_t(crop):
    cr = cv2.resize(crop, (PSZ, PSZ), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(cr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(((rgb - 0.5) / 0.5).transpose(2, 0, 1)))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args(); set_seed(0)
    truth = sorted(f[:-4] for f in os.listdir(TRUTH_DIR) if f.endswith(".png"))
    hd = set(map(pat, json.load(open("labels_handdraw/select.json"))["stems"]))
    e200 = set(pat(r["stem"]) for r in csv.DictReader(open("results/confirm200_ft_vs_zs.csv")))
    t12 = set(map(pat, json.load(open("labels/test_frozen.json"))["test"]))
    clean = [s for s in truth if pat(s) not in (hd | e200 | t12)]
    random.shuffle(clean); val_s = set(clean[:18])
    Xtr, Ytr, Xva, Yva = [], [], [], []
    print(f"[supcon] trích patch từ {len(clean)} ảnh...", flush=True)
    for s in clean:
        bgr = cv2.imread(f"{IMG_DIR}/{s}.jpg"); m = cv2.imread(f"{TRUTH_DIR}/{s}.png", 0) > 127
        spec, _, _ = clean_specimen(bgr); H, W = m.shape
        ys, xs = np.where(spec > 0)
        if len(ys) == 0: continue
        pos = np.argwhere(m);
        X = Xva if s in val_s else Xtr; Y = Yva if s in val_s else Ytr
        # tumor patches
        for _ in range(12):
            if len(pos) == 0: break
            cy, cx = pos[random.randint(0, len(pos)-1)]
            y = max(0, min(H-PSZ, cy-PSZ//2)); x = max(0, min(W-PSZ, cx-PSZ//2))
            if m[y:y+PSZ, x:x+PSZ].mean() > 0.3: X.append(patch_t(bgr[y:y+PSZ, x:x+PSZ])); Y.append(1)
        # healthy patches
        for _ in range(12):
            i = random.randint(0, len(ys)-1); cy, cx = ys[i], xs[i]
            y = max(0, min(H-PSZ, cy-PSZ//2)); x = max(0, min(W-PSZ, cx-PSZ//2))
            if m[y:y+PSZ, x:x+PSZ].mean() < 0.02: X.append(patch_t(bgr[y:y+PSZ, x:x+PSZ])); Y.append(0)
    Ytr = torch.tensor(Ytr); Yva = torch.tensor(Yva)
    print(f"[supcon] train={len(Xtr)} (u={int(Ytr.sum())}) val={len(Xva)} (u={int(Yva.sum())})", flush=True)
    net = Enc().to(DEVICE); opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    for ep in range(1, args.epochs+1):
        net.train(); idx = torch.randperm(len(Xtr)); tot=0; nb=0
        for i in range(0, len(idx), args.batch):
            b = idx[i:i+args.batch]
            if len(b) < 8: continue
            xb = torch.stack([Xtr[j] for j in b]).to(DEVICE); yb = Ytr[b].to(DEVICE)
            z, lg = net(xb)
            loss = supcon(z, yb) + 0.3 * F.binary_cross_entropy_with_logits(lg[:,0], yb.float())
            opt.zero_grad(); loss.backward(); opt.step(); tot+=float(loss); nb+=1
        if ep % 10 == 0:
            net.eval()
            with torch.no_grad():
                pr = torch.cat([torch.sigmoid(net(torch.stack(Xva[i:i+256]).to(DEVICE))[1][:,0]).cpu() for i in range(0,len(Xva),256)])
            yv = Yva.numpy(); pv = pr.numpy()
            # AUC
            from_pos = pv[yv==1]; from_neg = pv[yv==0]
            auc = np.mean([ (from_pos[:,None] > from_neg[None,:]).mean() ]) if len(from_pos) and len(from_neg) else 0
            print(f"[ep {ep}] loss={tot/max(nb,1):.4f} val_AUC(u/lành)={auc:.4f}", flush=True)
    torch.save({"net": net.state_dict()}, CKPT)
    # AUC cuối
    net.eval()
    with torch.no_grad():
        pr = torch.cat([torch.sigmoid(net(torch.stack(Xva[i:i+256]).to(DEVICE))[1][:,0]).cpu() for i in range(0,len(Xva),256)])
    yv = Yva.numpy(); pv = pr.numpy(); fp = pv[yv==1]; fn = pv[yv==0]
    auc = (fp[:,None] > fn[None,:]).mean()
    print(f"\n===== SUPCON: AUC phân biệt vân U/LÀNH = {auc:.4f} =====")
    print("=> AUC cao (>0.9) = vân tách được (có hi vọng localize) | ~0.7 = tường xác nhận")
    json.dump({"auc": float(auc)}, open("results/supcon_auc.json","w"), indent=1)
    print(f"-> ckpt {CKPT}, AUC {auc:.4f}", flush=True)

if __name__ == "__main__":
    main()
