"""
HƯỚNG MỚI (user 2026-06-26) — SELF-SUPERVISED PRETRAIN (MAE) trên ~1100 ảnh đại thể rồi
dùng encoder đó cho SEGMENT trên crop. Kiểm chứng: pretrain miền có giúp tách u không?
(Hoài nghi: tiền lệ "encoder lớn không là đòn bẩy"; SSL trên 1100 ảnh là ÍT data. DINOv2
from-scratch cần ~100k+ ảnh nên chọn MAE — bền hơn ở data nhỏ.)

Tự dựng SmallUNet (kiểm soát được encoder để nạp trọng số MAE — MONAI SegResNet không
nạp encoder ngoài tiện). 2 pha:
  --pretrain : MAE — che 60% block 32px của ảnh specimen, encoder->recon decoder dựng lại
               -> lưu encoder `checkpoints/mae_encoder.pt`. Pool = ảnh is_cut_surface.
  --finetune : nạp encoder MAE -> SmallUNet (encoder + seg decoder) -> train 318 patch crop
               -> eval 50 vẽ tay (full-auto/ceiling, so segR_rgb 0.879/0.616 & segR_lab 0.883).
               --scratch để train KHÔNG nạp MAE (đối chứng).
Env: /home/hvusynh2/conda_envs/medsam2_anno/bin/python.
"""
import argparse, csv, json, os, sys, random, hashlib
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import seg_crop as S
from seg_crop import (PatchSet, eval_handdraw, frag_boxes, set_seed, TRUTH_DIR, pat, make_channels, SIZE)
from specimen_clean import clean_specimen

DEVICE = "cuda"; IMG_DIR = "data/20241212"
ENC_CKPT = "checkpoints/mae_encoder.pt"

# ---------------- SmallUNet (encoder kiểm soát được) ----------------
def cblock(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.GroupNorm(8, o), nn.GELU(),
                         nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(8, o), nn.GELU())

class Encoder(nn.Module):
    """in_ch -> [e0@512(32), e1@256(64), e2@128(128), e3@64(256), e4@32(256)]."""
    def __init__(self, in_ch=3, w=(32, 64, 128, 256, 256)):
        super().__init__()
        self.stem = cblock(in_ch, w[0])
        self.d1 = cblock(w[0], w[1]); self.d2 = cblock(w[1], w[2])
        self.d3 = cblock(w[2], w[3]); self.d4 = cblock(w[3], w[4])
        self.pool = nn.MaxPool2d(2); self.w = w
    def forward(self, x):
        e0 = self.stem(x); e1 = self.d1(self.pool(e0)); e2 = self.d2(self.pool(e1))
        e3 = self.d3(self.pool(e2)); e4 = self.d4(self.pool(e3))
        return [e0, e1, e2, e3, e4]

class SegUNet(nn.Module):
    def __init__(self, in_ch=3, w=(32, 64, 128, 256, 256)):
        super().__init__()
        self.enc = Encoder(in_ch, w)
        self.u3 = cblock(w[4] + w[3], w[3]); self.u2 = cblock(w[3] + w[2], w[2])
        self.u1 = cblock(w[2] + w[1], w[1]); self.u0 = cblock(w[1] + w[0], w[0])
        self.out = nn.Conv2d(w[0], 1, 1)
    def up(self, x): return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    def forward(self, x):
        e0, e1, e2, e3, e4 = self.enc(x)
        d = self.u3(torch.cat([self.up(e4), e3], 1)); d = self.u2(torch.cat([self.up(d), e2], 1))
        d = self.u1(torch.cat([self.up(d), e1], 1)); d = self.u0(torch.cat([self.up(d), e0], 1))
        return self.out(d)

class MAERecon(nn.Module):
    """Encoder + decoder nhẹ dựng lại 3ch (không skip — ép bottleneck học ngữ nghĩa)."""
    def __init__(self, in_ch=3, w=(32, 64, 128, 256, 256)):
        super().__init__()
        self.enc = Encoder(in_ch, w)
        self.u3 = cblock(w[4], w[3]); self.u2 = cblock(w[3], w[2])
        self.u1 = cblock(w[2], w[1]); self.u0 = cblock(w[1], w[0])
        self.out = nn.Conv2d(w[0], 3, 1)
    def up(self, x): return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    def forward(self, x):
        e = self.enc(x)[-1]
        d = self.u3(self.up(e)); d = self.u2(self.up(d)); d = self.u1(self.up(d)); d = self.u0(self.up(d))
        return self.out(d)

# ---------------- MAE pretrain ----------------
class MAEData(torch.utils.data.Dataset):
    def __init__(self, stems): self.stems = stems
    def __len__(self): return len(self.stems)
    def __getitem__(self, i):
        bgr = cv2.imread(f"{IMG_DIR}/{self.stems[i]}.jpg")
        spec, _, _ = clean_specimen(bgr)
        ys, xs = np.where(spec > 0)
        if len(ys) > 100:
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            bgr = bgr[y0:y1 + 1, x0:x1 + 1]
        H, W = bgr.shape[:2]
        # random crop vuông rồi resize 512
        s = min(H, W); cy = random.randint(0, H - s); cx = random.randint(0, W - s)
        crop = cv2.resize(bgr[cy:cy + s, cx:cx + s], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        if random.random() < 0.5: crop = crop[:, ::-1]
        rgb = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(((rgb - 0.5) / 0.5).transpose(2, 0, 1))
        return img

def random_mask(B, ratio=0.6, block=32):
    g = SIZE // block
    m = (torch.rand(B, 1, g, g) < ratio).float()
    return F.interpolate(m, scale_factor=block, mode="nearest")  # (B,1,512,512) 1=che

def pretrain(args):
    set_seed(0)
    rows = list(csv.DictReader(open("processed/cut_surface_filter.csv")))
    pool = [r["stem"] for r in rows if str(r.get("is_cut_surface", "")).lower() in ("true", "1", "yes")]
    pool = [s for s in pool if os.path.isfile(f"{IMG_DIR}/{s}.jpg")]
    print(f"[mae] pool ảnh cut-surface = {len(pool)}", flush=True)
    dl = torch.utils.data.DataLoader(MAEData(pool), batch_size=args.batch, shuffle=True,
                                     num_workers=6, drop_last=True)
    model = MAERecon(3).to(DEVICE)
    print(f"[mae] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0; nb = 0
        for img in dl:
            img = img.to(DEVICE); m = random_mask(img.size(0)).to(DEVICE)
            masked = img * (1 - m)
            rec = model(masked)
            loss = (((rec - img) ** 2) * m).sum() / (m.sum() * 3 + 1e-6)  # MSE trên vùng che
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        sched.step()
        print(f"[mae ep {ep}] recon_mse={tot/max(nb,1):.4f}", flush=True)
    torch.save({"enc": model.enc.state_dict()}, ENC_CKPT)
    print(f"[mae] lưu encoder -> {ENC_CKPT}", flush=True)

# ---------------- finetune segment ----------------
def finetune(args):
    set_seed(0); tag = args.tag or ("seg_mae" if not args.scratch else "seg_unet_scratch")
    ckpt_out = f"checkpoints/seg_crop_{tag}.pt"
    net = SegUNet(3).to(DEVICE)
    if not args.scratch:
        enc = torch.load(ENC_CKPT, weights_only=False)["enc"]
        net.enc.load_state_dict(enc); print(f"[ft] nạp encoder MAE từ {ENC_CKPT}", flush=True)
    else:
        print("[ft] SCRATCH — không nạp MAE (đối chứng)", flush=True)
    print(f"[ft] tag={tag} params={sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)

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
    print(f"[split] train patch={len(items_tr)} val patch={len(items_va)}", flush=True)
    tr = torch.utils.data.DataLoader(PatchSet(items_tr, "rgb", True), batch_size=args.batch,
                                     shuffle=True, num_workers=4, drop_last=True)
    va = torch.utils.data.DataLoader(PatchSet(items_va, "rgb", False), batch_size=args.batch,
                                     shuffle=False, num_workers=2)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    @torch.no_grad()
    def val_dice(thr=0.5):
        net.eval(); ds = []
        for x, t, _ in va:
            p = (torch.sigmoid(net(x.to(DEVICE))) > thr).cpu().numpy(); g = t.numpy() > 0.5
            for k in range(len(p)):
                sden = p[k].sum() + g[k].sum(); ds.append(1.0 if sden == 0 else 2 * (p[k] & g[k]).sum() / sden)
        return float(np.mean(ds)) if ds else 0.0

    best = -1.0
    for ep in range(1, args.epochs + 1):
        net.train(); tot = 0.0; nb = 0
        for x, t, _ in tr:
            x = x.to(DEVICE); t = t.to(DEVICE); lg = net(x)
            loss = S.soft_dice(lg, t) + F.binary_cross_entropy_with_logits(lg.float(), t.float())
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        sched.step(); vd = val_dice(); star = ""
        if vd > best:
            best = vd; torch.save({"net": net.state_dict(), "arch": "segunet", "channels": "rgb",
                                   "val_dice": vd, "epoch": ep}, ckpt_out); star = "  *best"
        print(f"[ep {ep}] loss={tot/max(nb,1):.4f} val_dice={vd:.4f}{star}", flush=True)
    print(f"\nTrain xong. best val={best:.4f} -> {ckpt_out}", flush=True)
    ck = torch.load(ckpt_out, weights_only=False); net.load_state_dict(ck["net"])
    eval_handdraw(net, "rgb", tag)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain", action="store_true"); ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--scratch", action="store_true"); ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()
    if args.pretrain: pretrain(args)
    if args.finetune: finetune(args)
    if not (args.pretrain or args.finetune): ap.error("cần --pretrain và/hoặc --finetune")

if __name__ == "__main__":
    main()
