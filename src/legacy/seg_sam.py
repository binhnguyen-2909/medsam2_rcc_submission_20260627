"""
HƯỚNG A — SEGMENTER CHUYÊN DỤNG (KHÔNG box-prompt). Head U-Net dày trên đặc trưng đa tầng
của SAM encoder (đông cứng) -> dự đoán DENSE mask u trực tiếp từ ảnh. Học u theo kết cấu/màu,
bỏ hẳn nút thắt box->SAM (spill). Train trên 178 mask THẬT, giữ 50 ảnh vẽ tay làm test sạch.

Feature lấy từ cache_feats/{stem}.pt (feat 256x64x64 + hrf [(32,256,256),(64,128,128)]),
không có thì tự chạy SAM encoder. Head out logits 256x256 -> upsample full-res.
  python seg_sam.py --epochs 60 --batch 16
Lưu best val Dice -> checkpoints/seg_sam.pt ; cuối: eval 50 ảnh vẽ tay (so 0.666/0.857).
"""
import argparse, csv, json, os, sys, random, hashlib
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

IMG_DIR = "data/20241212"; TRUTH_DIR = "labels_truth/masks"; SAM_DIR = "labels/masks"
HMASK = "labels_handdraw/masks"
# Backbone chọn qua env (mặc định tiny). LARGE: SAM2_CONFIG=configs/sam2.1_hiera_l SAM2_CKPT=checkpoints/sam2.1_hiera_large.pt
CFG = os.environ.get("SAM2_CONFIG", "configs/sam2.1_hiera_t512")
CKPT = os.environ.get("SAM2_CKPT", "checkpoints/sam2.1_hiera_tiny.pt")
RES = int(os.environ.get("SAM2_RES", "1024"))
# Cache riêng theo backbone để feature tiny/large không lẫn nhau.
CACHE = os.environ.get("SEG_CACHE", "cache_feats_large" if "large" in CKPT else "cache_feats")
DEVICE = "cuda"; AC = torch.autocast("cuda", dtype=torch.bfloat16); MIN_FRAC = 0.002
def pat(s): return s.split("^")[0]
def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)

def truth_path(s):
    tp = f"{TRUTH_DIR}/{s}.png"; return tp if os.path.isfile(tp) else f"{SAM_DIR}/{s}.png"
def read_mask256(s, src=None):
    m = cv2.imread(src or truth_path(s), 0)
    return (cv2.resize((m > 127).astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA) > 0.5).astype(np.float32)
def ncomp(m):
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return max(1, int((st[1:, cv2.CC_STAT_AREA] >= MIN_FRAC * m.size).sum()))

class SegHead(nn.Module):
    """U-Net decoder trên feat(256,64,64) + hrf0(32,256,256) + hrf1(64,128,128) -> logits 256x256."""
    def __init__(self):
        super().__init__()
        def cb(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU(),
                                           nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())
        self.c3 = cb(256, 128)               # 64x64
        self.c2 = cb(128 + 64, 64)           # 128x128
        self.c1 = cb(64 + 32, 32)            # 256x256
        self.out = nn.Conv2d(32, 1, 1)
    def forward(self, feat, hrf0, hrf1):
        x = self.c3(feat)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 128
        x = self.c2(torch.cat([x, hrf1], 1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 256
        x = self.c1(torch.cat([x, hrf0], 1))
        return self.out(x)                   # (B,1,256,256)

def dice_bce(logits, target):
    p = torch.sigmoid(logits.float()); t = target.float(); dims = (1, 2, 3)
    inter = (p * t).sum(dims); d = 1 - (2 * inter + 1) / (p.sum(dims) + t.sum(dims) + 1)
    bce = F.binary_cross_entropy_with_logits(logits.float(), t, reduction="none").mean(dims)
    return (d + bce).mean()

_predictor = None
def get_feats(stem):
    """feat,hrf0,hrf1 (CPU fp32). Ưu tiên cache; thiếu thì chạy SAM encoder."""
    cp = f"{CACHE}/{stem}.pt"
    if os.path.isfile(cp):
        d = torch.load(cp, map_location="cpu", weights_only=False)
        return d["feat"].float(), d["hrf"][0].float(), d["hrf"][1].float()
    global _predictor
    if _predictor is None:
        model = build_sam2(CFG, CKPT, device=DEVICE, hydra_overrides_extra=[f"++model.image_size={RES}"])
        for p in model.parameters(): p.requires_grad_(False)
        _predictor = SAM2ImagePredictor(model)
    rgb = cv2.cvtColor(cv2.imread(f"{IMG_DIR}/{stem}.jpg"), cv2.COLOR_BGR2RGB)
    with torch.inference_mode(), AC: _predictor.set_image(rgb)
    fe = _predictor._features["image_embed"][0].float().cpu()
    hr = [h[0].float().cpu() for h in _predictor._features["high_res_feats"]]
    return fe, hr[0], hr[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--n_val", type=int, default=12)
    ap.add_argument("--ckpt_out", default="checkpoints/seg_sam.pt")
    args = ap.parse_args(); set_seed(0)

    truth = sorted(f[:-4] for f in os.listdir(TRUTH_DIR) if f.endswith(".png"))
    hd = set(map(pat, json.load(open("labels_handdraw/select.json"))["stems"]))
    e200 = set(pat(r["stem"]) for r in csv.DictReader(open("results/confirm200_ft_vs_zs.csv")))
    t12 = set(map(pat, json.load(open("labels/test_frozen.json"))["test"]))
    excl = hd | e200 | t12
    clean = [s for s in truth if pat(s) not in excl]
    clean.sort(key=lambda s: hashlib.md5(s.encode()).hexdigest())
    val = clean[:args.n_val]; train = clean[args.n_val:]
    print(f"[split] seg train={len(train)} val={len(val)} (handdraw 50 HELD-OUT, 0 rò rỉ)", flush=True)

    # nạp feature + target vào RAM (nhỏ)
    def load_set(stems):
        out = []
        for s in stems:
            fe, h0, h1 = get_feats(s)
            out.append((fe, h0, h1, torch.from_numpy(read_mask256(s))[None]))
        return out
    print("Nạp feature train...", flush=True); tr = load_set(train)
    print("Nạp feature val...", flush=True); va = load_set(val)

    net = SegHead().to(DEVICE)
    print(f"Tham số head: {sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)

    @torch.no_grad()
    def val_dice(thr=0.5):
        net.eval(); ds = []
        for fe, h0, h1, t in va:
            lg = net(fe[None].to(DEVICE), h0[None].to(DEVICE), h1[None].to(DEVICE))
            p = (torch.sigmoid(lg)[0, 0] > thr).cpu().numpy(); g = t[0].numpy() > 0.5
            s = p.sum() + g.sum(); ds.append(1.0 if s == 0 else 2 * (p & g).sum() / s)
        return float(np.mean(ds))

    logf = open("results/seg_sam_log.csv", "w", newline=""); logw = csv.writer(logf)
    logw.writerow(["epoch", "loss", "val_dice"]); logf.flush()
    best = -1.0; idx = list(range(len(tr)))
    for ep in range(1, args.epochs + 1):
        net.train(); random.shuffle(idx); tot = 0.0; nb = 0
        for i in range(0, len(idx), args.batch):
            bi = idx[i:i + args.batch]
            fe = torch.stack([tr[j][0] for j in bi]).to(DEVICE)
            h0 = torch.stack([tr[j][1] for j in bi]).to(DEVICE)
            h1 = torch.stack([tr[j][2] for j in bi]).to(DEVICE)
            tg = torch.stack([tr[j][3] for j in bi]).to(DEVICE)
            lg = net(fe, h0, h1); loss = dice_bce(lg, tg)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        vd = val_dice(); star = ""
        if vd > best:
            best = vd; torch.save({"net": net.state_dict(), "val_dice": vd, "epoch": ep}, args.ckpt_out); star = "  *best"
        print(f"[ep {ep}] loss={tot/max(nb,1):.4f} | val_dice={vd:.4f}{star}", flush=True)
        logw.writerow([ep, round(tot/max(nb,1), 4), round(vd, 4)]); logf.flush()
    logf.close()
    print(f"\nTrain xong. best val={best:.4f} -> {args.ckpt_out}", flush=True)

    # ---- EVAL 50 ẢNH VẼ TAY (full-res, so 0.666/0.857) ----
    ck = torch.load(args.ckpt_out, weights_only=False); net.load_state_dict(ck["net"]); net.eval()
    stems = json.load(open("labels_handdraw/select.json"))["stems"]
    have = [s for s in stems if os.path.isfile(f"{HMASK}/{s}.png")]
    rows = []
    for s in have:
        gt = cv2.imread(f"{HMASK}/{s}.png", 0) > 127; H, W = gt.shape
        fe, h0, h1 = get_feats(s)
        with torch.no_grad():
            lg = net(fe[None].to(DEVICE), h0[None].to(DEVICE), h1[None].to(DEVICE))
            pm = torch.sigmoid(lg)[0, 0].cpu().numpy()
        pm = cv2.resize(pm, (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
        sden = pm.sum() + gt.sum(); d = 1.0 if sden == 0 else 2 * (pm & gt).sum() / sden
        rows.append((ncomp(gt), float(d)))
        print(f"  {s[:24]:26} n={ncomp(gt)} seg={d:.3f}", flush=True)
    import statistics as st
    def med(f=lambda r: True):
        v = [r[1] for r in rows if f(r)]; return st.median(v), st.mean(v), len(v)
    a = med(); o = med(lambda r: r[0] <= 1); m = med(lambda r: r[0] > 1)
    print(f"\n===== SEGMENTER (SAM-feat + head) trên 50 vẽ tay =====")
    print(f"seg median={a[0]:.4f} mean={a[1]:.4f} (n={a[2]}) | 1u={o[0]:.3f} (n={o[2]}) >1u={m[0]:.3f} (n={m[2]})")
    print(f"backbone={os.path.basename(CKPT)}  cache={CACHE}", flush=True)
    print(f"SO: seg-tiny=0.624 | full-auto(box+SAM)=0.666 | ceiling=0.857 | nhãn SAM cũ=0.554", flush=True)
    tag = "_large" if "large" in CKPT else ""
    json.dump({"median": a[0], "mean": a[1], "1u": o[0], ">1u": m[0], "backbone": os.path.basename(CKPT)},
              open(f"results/seg_sam_handdraw{tag}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
