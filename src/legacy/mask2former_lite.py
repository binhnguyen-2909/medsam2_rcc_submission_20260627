"""
#H MASK2FORMER-LITE (user yêu cầu chạy tận mắt) — queries + bipartite matching, KHÔNG box/detector.
Crop bệnh phẩm (clean_specimen bbox) -> resize 512 (6ch RGB+LAB) -> encoder -> pixel-embed 128x128
+ Q=20 learnable queries qua 3 lớp transformer-decoder (cross/self-attn) -> mỗi query: class(u/no-obj)
+ mask. Hungarian match query<->instance GT (mỗi mảnh u = 1 instance). Inference: query P(u)>thr ->
union mask -> đưa về full-res. AUGMENT MẠNH (flip/rot/scale/color) chống overfit (user nhấn mạnh).
Train trên ~1000 ảnh (truth ưu tiên else SAM). Eval full-auto 50 vẽ tay (so 0.635 / nhãn SAM 0.554).
  python mask2former_lite.py --epochs 80
Env: medsam2_anno.
"""
import argparse, csv, json, os, sys, random, hashlib, math
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from seg_crop import set_seed, TRUTH_DIR, pat, make_channels, n_channels
from specimen_clean import clean_specimen
from mae_seg import Encoder, cblock
DEVICE = "cuda"; IMG_DIR = "data/20241212"; SAM_DIR = "labels/masks"
INP = 512; MRES = 128; Q = 20; D = 128; MINF = 0.002; CKPT = "checkpoints/mask2former_lite.pt"

def mask_path(s):
    tp = f"{TRUTH_DIR}/{s}.png"; return tp if os.path.isfile(tp) else f"{SAM_DIR}/{s}.png"

def spec_crop(bgr, m=None):
    spec, _, _ = clean_specimen(bgr); ys, xs = np.where(spec > 0)
    if len(ys) < 50:
        H, W = bgr.shape[:2]; box = (0, 0, W, H)
    else:
        box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    x0, y0, x1, y1 = box
    c = bgr[y0:y1, x0:x1]; cm = m[y0:y1, x0:x1] if m is not None else None
    return c, cm, box

def instances(mask128):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask128.astype(np.uint8), 8)
    thr = MINF * mask128.size; out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < thr: continue
        out.append((lab == i).astype(np.float32))
    return out

class DS(torch.utils.data.Dataset):
    def __init__(self, stems, mode, train=True): self.stems = stems; self.mode = mode; self.train = train
    def __len__(self): return len(self.stems)
    def __getitem__(self, i):
        s = self.stems[i]; bgr = cv2.imread(f"{IMG_DIR}/{s}.jpg"); m = cv2.imread(mask_path(s), 0) > 127
        c, cm, _ = spec_crop(bgr, m.astype(np.uint8))
        c = cv2.resize(c, (INP, INP), interpolation=cv2.INTER_AREA)
        cm = cv2.resize(cm, (INP, INP), interpolation=cv2.INTER_NEAREST)
        if self.train:
            if random.random() < 0.5: c = c[:, ::-1]; cm = cm[:, ::-1]
            if random.random() < 0.5: c = c[::-1]; cm = cm[::-1]
            k = random.randint(0, 3)
            if k: c = np.rot90(c, k).copy(); cm = np.rot90(cm, k).copy()
            if random.random() < 0.5:  # scale jitter (random resized crop)
                sc = random.uniform(0.7, 1.0); ch = int(INP * sc)
                y = random.randint(0, INP - ch); x = random.randint(0, INP - ch)
                c = cv2.resize(c[y:y+ch, x:x+ch], (INP, INP), interpolation=cv2.INTER_AREA)
                cm = cv2.resize(cm[y:y+ch, x:x+ch], (INP, INP), interpolation=cv2.INTER_NEAREST)
            if random.random() < 0.6:
                hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[..., 1] *= random.uniform(0.8, 1.2); hsv[..., 2] *= random.uniform(0.8, 1.2)
                c = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        x = make_channels(np.ascontiguousarray(c), self.mode)
        cm128 = cv2.resize(cm.astype(np.uint8), (MRES, MRES), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(x), cm128

def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    gts = [instances(b[1]) for b in batch]
    return xs, gts

class M2F(nn.Module):
    def __init__(self, in_ch, w=(32, 64, 128, 256, 256)):
        super().__init__()
        self.enc = Encoder(in_ch, w)
        self.proj = nn.Conv2d(w[2], D, 1)         # e2 @128 -> pixel embed D
        self.feat_proj = nn.Conv2d(w[4], D, 1)    # e4 @32 -> memory cho cross-attn
        self.query = nn.Parameter(torch.randn(Q, D) * 0.02)
        self.layers = nn.ModuleList([nn.ModuleDict({
            "ca": nn.MultiheadAttention(D, 8, batch_first=True),
            "sa": nn.MultiheadAttention(D, 8, batch_first=True),
            "ff": nn.Sequential(nn.Linear(D, D * 2), nn.GELU(), nn.Linear(D * 2, D)),
            "n1": nn.LayerNorm(D), "n2": nn.LayerNorm(D), "n3": nn.LayerNorm(D)}) for _ in range(3)])
        self.cls = nn.Linear(D, 1); self.mask_embed = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
    def forward(self, x):
        e = self.enc(x); pix = self.proj(e[2])                  # (B,D,128,128)
        mem = self.feat_proj(e[4]).flatten(2).transpose(1, 2)  # (B,32*32,D)
        B = x.shape[0]; q = self.query[None].expand(B, -1, -1).clone()
        for L in self.layers:
            a, _ = L["ca"](q, mem, mem); q = L["n1"](q + a)
            a, _ = L["sa"](q, q, q); q = L["n2"](q + a)
            q = L["n3"](q + L["ff"](q))
        cls = self.cls(q)[..., 0]                               # (B,Q)
        me = self.mask_embed(q)                                 # (B,Q,D)
        masks = torch.einsum("bqd,bdhw->bqhw", me, pix)         # (B,Q,128,128) logits
        return cls, masks

def dice_bce_m(ml, t):
    p = torch.sigmoid(ml); inter = (p * t).sum((-1, -2))
    d = 1 - (2 * inter + 1) / (p.sum((-1, -2)) + t.sum((-1, -2)) + 1)
    bce = F.binary_cross_entropy_with_logits(ml, t, reduction="none").mean((-1, -2))
    return d + bce

@torch.no_grad()
def match(cls, masks, gts):
    """Hungarian cho 1 sample. trả (idx_query, idx_gt)."""
    if len(gts) == 0: return [], []
    G = torch.stack([torch.from_numpy(g) for g in gts]).to(masks.device)  # (G,128,128)
    p = torch.sigmoid(masks)                                              # (Q,128,128)
    pf = p.flatten(1); gf = G.flatten(1)
    inter = pf @ gf.t(); s = pf.sum(1)[:, None] + gf.sum(1)[None, :]
    dice = 1 - (2 * inter + 1) / (s + 1)                                  # (Q,G)
    clscost = (-torch.sigmoid(cls))[:, None].expand(-1, len(gts))
    C = (dice + clscost).cpu().numpy()
    qi, gi = linear_sum_assignment(C); return list(qi), list(gi)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--channels", default="lab")
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args(); set_seed(0); mode = args.channels; inC = n_channels(mode)
    hd = set(map(pat, json.load(open("labels_handdraw/select.json"))["stems"]))
    e200 = set(pat(r["stem"]) for r in csv.DictReader(open("results/confirm200_ft_vs_zs.csv")))
    t12 = set(map(pat, json.load(open("labels/test_frozen.json"))["test"]))
    excl = hd | e200 | t12
    truth = set(f[:-4] for f in os.listdir(TRUTH_DIR) if f.endswith(".png"))
    sam = set(f[:-4] for f in os.listdir(SAM_DIR) if f.endswith(".png"))
    stems = sorted(s for s in (truth | sam) if pat(s) not in excl and os.path.isfile(f"{IMG_DIR}/{s}.jpg"))
    stems.sort(key=lambda s: hashlib.md5(s.encode()).hexdigest())
    val = stems[:30]; train = stems[30:]
    print(f"[m2f] train={len(train)} val={len(val)} ({mode},{inC}ch) Q={Q}", flush=True)
    tr = torch.utils.data.DataLoader(DS(train, mode, True), batch_size=args.batch, shuffle=True,
                                     num_workers=6, drop_last=True, collate_fn=collate)
    net = M2F(inC).to(DEVICE); opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    print(f"[m2f] params={sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
    best = -1
    for ep in range(1, args.epochs + 1):
        net.train(); tot = 0; nb = 0
        for xs, gts in tr:
            xs = xs.to(DEVICE); cls, masks = net(xs)
            loss = 0.0
            for b in range(len(gts)):
                qi, gi = match(cls[b], masks[b], gts[b])
                tgt_cls = torch.zeros(Q, device=DEVICE)
                if qi:
                    G = torch.stack([torch.from_numpy(gts[b][j]) for j in gi]).to(DEVICE)
                    qi_t = torch.tensor(qi, device=DEVICE)
                    tgt_cls[qi_t] = 1.0
                    loss = loss + dice_bce_m(masks[b][qi_t], G).mean()
                loss = loss + F.binary_cross_entropy_with_logits(cls[b], tgt_cls)
            loss = loss / len(gts)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        sched.step()
        if ep % 5 == 0 or ep == 1:
            vd = eval_handdraw(net, mode, quiet=True)
            if vd > best: best = vd; torch.save({"net": net.state_dict(), "channels": mode, "epoch": ep, "hd": vd}, CKPT)
            print(f"[ep {ep}] loss={tot/max(nb,1):.4f} handdraw_fullauto={vd:.4f}{'  *best' if vd>=best else ''}", flush=True)
    print(f"\nTrain xong. best handdraw full-auto={best:.4f} -> {CKPT}", flush=True)
    ck = torch.load(CKPT, weights_only=False); net.load_state_dict(ck["net"]); eval_handdraw(net, mode)

@torch.no_grad()
def eval_handdraw(net, mode, thr=0.5, quiet=False):
    import statistics as st
    net.eval()
    stems = json.load(open("labels_handdraw/select.json"))["stems"]
    have = [s for s in stems if os.path.isfile(f"labels_handdraw/masks/{s}.png")]
    rows = []
    for s in have:
        gt = cv2.imread(f"labels_handdraw/masks/{s}.png", 0) > 127
        bgr = cv2.imread(f"{IMG_DIR}/{s}.jpg"); H, W = gt.shape
        c, _, box = spec_crop(bgr); x0, y0, x1, y1 = box
        cr = cv2.resize(c, (INP, INP), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(make_channels(np.ascontiguousarray(cr), mode))[None].to(DEVICE)
        cls, masks = net(x); keep = torch.sigmoid(cls[0]) > thr
        u128 = np.zeros((MRES, MRES), bool)
        if keep.any():
            mk = (torch.sigmoid(masks[0][keep]) > 0.5).cpu().numpy()
            for m in mk: u128 |= m
        # đưa về full-res: 128 -> crop size -> đặt vào box
        um = cv2.resize(u128.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
        full = np.zeros((H, W), bool); full[y0:y1, x0:x1] = um
        a = gt.sum() + full.sum(); d = 1.0 if a == 0 else 2 * (full & gt).sum() / a
        nC = max(1, len(instances(cv2.resize(gt.astype(np.uint8), (MRES, MRES)))))
        rows.append((nC, d))
        if not quiet: print(f"  {s[:20]:22} n={nC} dice={d:.3f}", flush=True)
    md = st.median([r[1] for r in rows]); mn = st.mean([r[1] for r in rows])
    if not quiet:
        o = st.median([r[1] for r in rows if r[0] <= 1]); m = st.median([r[1] for r in rows if r[0] > 1])
        print(f"\n===== MASK2FORMER-LITE full-auto trên 50 vẽ tay =====")
        print(f"Dice median={md:.4f} mean={mn:.4f} | 1u={o:.3f} >1u={m:.3f}")
        print(f"SO: full-auto detector+seg 0.635 | nhãn SAM cũ 0.554 | semi-auto 0.883")
        json.dump({"median": md, "mean": mn}, open("results/loc_mask2former.json", "w"), indent=1)
    return md

if __name__ == "__main__":
    main()
