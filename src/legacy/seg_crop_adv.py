"""
HƯỚNG MỚI (user 2026-06-26) — chống SPILL viền bằng (A) ADVERSARIAL + (B) clDice.
Nền: champion segR_lab (SegResNet+LAB trên crop, ceiling 0.883 > SAM 0.857). Còn cách
red-ceiling 0.938 ~0.055 = spill viền (mask tràn ra mô lành). Hai vũ khí:

(A) ADVERSARIAL (PatchGAN): D phân biệt (crop, mask THẬT vẽ tay/truth)=1 vs (crop, mask
    Generator đoán)=0. G bị ép co viền cho giống nét người → đánh lừa D. LSGAN cho ổn định.
(B) clDice (soft centerline Dice): phạt mask nứt/thủng/mọc nhánh. (Hoài nghi cho blob,
    thử kiểm chứng.)

Dùng lại hạ tầng seg_crop (PatchSet, make_channels, build_model, eval_handdraw, split).
  python seg_crop_adv.py --adv --tag segR_lab_adv
  python seg_crop_adv.py --cldice --tag segR_lab_cldice
  python seg_crop_adv.py --adv --cldice --tag segR_lab_adv_cldice
Env: /home/hvusynh2/conda_envs/medsam2_anno/bin/python.
"""
import argparse, csv, json, os, sys, random, hashlib
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import seg_crop as S
from seg_crop import (PatchSet, build_model, n_channels, soft_dice, eval_handdraw,
                      frag_boxes, set_seed, TRUTH_DIR, pat)

DEVICE = "cuda"

# ---------------- clDice ----------------
def soft_erode(x): return -F.max_pool2d(-x, 3, 1, 1)
def soft_dilate(x): return F.max_pool2d(x, 3, 1, 1)
def soft_open(x): return soft_dilate(soft_erode(x))
def soft_skel(x, iters=10):
    x1 = soft_open(x); skel = F.relu(x - x1)
    for _ in range(iters):
        x = soft_erode(x); x1 = soft_open(x)
        d = F.relu(x - x1); skel = skel + F.relu(d - skel * d)
    return skel
def cldice_loss(p, t, iters=10):
    """p,t: (B,1,H,W) trong [0,1]. -> 1 - clDice."""
    sp = soft_skel(p, iters); st = soft_skel(t, iters)
    tprec = (sp * t).sum() / (sp.sum() + 1e-6)
    tsens = (st * p).sum() / (st.sum() + 1e-6)
    return 1 - 2 * tprec * tsens / (tprec + tsens + 1e-6)

# ---------------- PatchGAN Discriminator ----------------
class PatchD(nn.Module):
    """Nhận (channels ảnh + mask) -> bản đồ real/fake (PatchGAN). LSGAN."""
    def __init__(self, in_ch):
        super().__init__()
        def blk(i, o, s, norm=True):
            L = [nn.Conv2d(i, o, 4, s, 1)]
            if norm: L.append(nn.InstanceNorm2d(o))
            L.append(nn.LeakyReLU(0.2, True)); return L
        self.net = nn.Sequential(
            *blk(in_ch, 64, 2, norm=False), *blk(64, 128, 2), *blk(128, 256, 2),
            *blk(256, 256, 2), nn.Conv2d(256, 1, 4, 1, 1))
    def forward(self, x): return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="segresnet"); ap.add_argument("--channels", default="lab")
    ap.add_argument("--adv", action="store_true"); ap.add_argument("--cldice", action="store_true")
    ap.add_argument("--lambda_adv", type=float, default=0.05)
    ap.add_argument("--lambda_cl", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--n_val", type=int, default=28); ap.add_argument("--tag", default=None)
    ap.add_argument("--ckpt_out", default=None)
    args = ap.parse_args(); set_seed(0)
    tag = args.tag or f"{args.arch}_{args.channels}" + ("_adv" if args.adv else "") + ("_cldice" if args.cldice else "")
    ckpt_out = args.ckpt_out or f"checkpoints/seg_crop_{tag}.pt"
    inC = n_channels(args.channels)
    print(f"[cfg] {tag}: arch={args.arch} ch={args.channels}({inC}) adv={args.adv}(λ{args.lambda_adv}) "
          f"cldice={args.cldice}(λ{args.lambda_cl})", flush=True)

    # split sạch (giống seg_crop)
    truth = sorted(f[:-4] for f in os.listdir(TRUTH_DIR) if f.endswith(".png"))
    hd = set(map(pat, json.load(open("labels_handdraw/select.json"))["stems"]))
    e200 = set(pat(r["stem"]) for r in csv.DictReader(open("results/confirm200_ft_vs_zs.csv")))
    t12 = set(map(pat, json.load(open("labels/test_frozen.json"))["test"]))
    excl = hd | e200 | t12
    clean = [s for s in truth if pat(s) not in excl]
    clean.sort(key=lambda s: hashlib.md5(s.encode()).hexdigest())
    val_stems = set(clean[:args.n_val // 2])
    items_tr, items_va = [], []
    for s in clean:
        m = cv2.imread(f"{TRUTH_DIR}/{s}.png", 0) > 127
        for b in frag_boxes(m):
            (items_va if s in val_stems else items_tr).append((s, b))
    print(f"[split] train patch={len(items_tr)} val patch={len(items_va)} (50 vẽ tay HELD-OUT)", flush=True)

    tr = torch.utils.data.DataLoader(PatchSet(items_tr, args.channels, True),
                                     batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    va = torch.utils.data.DataLoader(PatchSet(items_va, args.channels, False),
                                     batch_size=args.batch, shuffle=False, num_workers=2)
    net = build_model(args.arch, inC).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    print(f"[model] G params={sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)

    D = optD = None
    if args.adv:
        D = PatchD(inC + 1).to(DEVICE)
        optD = torch.optim.AdamW(D.parameters(), lr=args.lr, weight_decay=args.wd)
        print(f"[model] D params={sum(p.numel() for p in D.parameters())/1e6:.2f}M (PatchGAN, LSGAN)", flush=True)

    @torch.no_grad()
    def val_dice(thr=0.5):
        net.eval(); ds = []
        for x, t, _ in va:
            p = (torch.sigmoid(net(x.to(DEVICE))) > thr).cpu().numpy(); g = t.numpy() > 0.5
            for k in range(len(p)):
                sden = p[k].sum() + g[k].sum(); ds.append(1.0 if sden == 0 else 2 * (p[k] & g[k]).sum() / sden)
        return float(np.mean(ds)) if ds else 0.0

    logf = open(f"results/seg_crop_{tag}_log.csv", "w", newline=""); logw = csv.writer(logf)
    logw.writerow(["epoch", "loss_g", "loss_d", "val_dice"]); logf.flush()
    best = -1.0
    for ep in range(1, args.epochs + 1):
        net.train(); tg = td = 0.0; nb = 0
        for x, t, _ in tr:
            x = x.to(DEVICE); t = t.to(DEVICE)
            lg = net(x); p = torch.sigmoid(lg)
            # --- D step ---
            ld_val = 0.0
            if args.adv:
                D.train()
                real = D(torch.cat([x, t], 1)); fake = D(torch.cat([x, p.detach()], 1))
                lossD = 0.5 * (F.mse_loss(real, torch.ones_like(real)) +
                               F.mse_loss(fake, torch.zeros_like(fake)))
                optD.zero_grad(); lossD.backward(); optD.step(); ld_val = float(lossD)
            # --- G step ---
            loss = soft_dice(lg, t) + F.binary_cross_entropy_with_logits(lg.float(), t.float())
            if args.cldice:
                loss = loss + args.lambda_cl * cldice_loss(p, t)
            if args.adv:
                adv = D(torch.cat([x, p], 1))
                loss = loss + args.lambda_adv * F.mse_loss(adv, torch.ones_like(adv))
            opt.zero_grad(); loss.backward(); opt.step()
            tg += float(loss); td += ld_val; nb += 1
        sched.step()
        vd = val_dice(); star = ""
        if vd > best:
            best = vd; torch.save({"net": net.state_dict(), "arch": args.arch, "channels": args.channels,
                                   "val_dice": vd, "epoch": ep}, ckpt_out); star = "  *best"
        print(f"[ep {ep}] loss_g={tg/max(nb,1):.4f} loss_d={td/max(nb,1):.4f} val_dice={vd:.4f}{star}", flush=True)
        logw.writerow([ep, round(tg/max(nb,1), 4), round(td/max(nb,1), 4), round(vd, 4)]); logf.flush()
    logf.close()
    print(f"\nTrain xong. best val(patch)={best:.4f} -> {ckpt_out}", flush=True)
    ck = torch.load(ckpt_out, weights_only=False); net.load_state_dict(ck["net"])
    eval_handdraw(net, args.channels, tag)

if __name__ == "__main__":
    main()
