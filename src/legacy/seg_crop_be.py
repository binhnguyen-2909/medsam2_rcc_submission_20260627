"""
HƯỚNG MỚI (user 2026-06-26) — BODY-EDGE DECOUPLING (chia để trị).
2 đầu ra: Head Body học LÕI u (nhãn = mask bị erode), Head Edge học VÀNH viền (nhãn = vành
mỏng quanh u). Module fuse hợp lõi+viền -> mask cuối. Tránh "pha loãng" tập trung khi học,
ép viền chính xác hơn. Train crop, eval 50 vẽ tay.
  python seg_crop_be.py --channels lab --tag be_lab --epochs 60
Env: /home/hvusynh2/conda_envs/medsam2_anno/bin/python.
"""
import argparse, csv, json, os, sys, hashlib
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import seg_crop as S
from seg_crop import (PatchSet, n_channels, soft_dice, eval_handdraw, frag_boxes, set_seed,
                      TRUTH_DIR, pat)
from mae_seg import Encoder, cblock
DEVICE = "cuda"

class BodyEdgeUNet(nn.Module):
    def __init__(self, in_ch, w=(32, 64, 128, 256, 256)):
        super().__init__()
        self.enc = Encoder(in_ch, w)
        self.u3 = cblock(w[4] + w[3], w[3]); self.u2 = cblock(w[3] + w[2], w[2])
        self.u1 = cblock(w[2] + w[1], w[1]); self.u0 = cblock(w[1] + w[0], w[0])
        self.body = nn.Conv2d(w[0], 1, 1); self.edge = nn.Conv2d(w[0], 1, 1)
        self.fuse = nn.Sequential(nn.Conv2d(2, 16, 3, 1, 1), nn.GELU(), nn.Conv2d(16, 1, 1))
    def up(self, z): return F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False)
    def feats(self, x):
        e0, e1, e2, e3, e4 = self.enc(x)
        d = self.u3(torch.cat([self.up(e4), e3], 1)); d = self.u2(torch.cat([self.up(d), e2], 1))
        d = self.u1(torch.cat([self.up(d), e1], 1)); d = self.u0(torch.cat([self.up(d), e0], 1))
        return d
    def forward_train(self, x):
        d = self.feats(x); b = self.body(d); e = self.edge(d)
        final = self.fuse(torch.cat([b, e], 1))
        return final, b, e
    def forward(self, x):  # eval: chỉ cần mask cuối
        return self.forward_train(x)[0]

def erode(t, k=9):
    return -F.max_pool2d(-t, k, 1, k // 2)

def dbce(lg, tg):
    return soft_dice(lg, tg) + F.binary_cross_entropy_with_logits(lg.float(), tg.float())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="lab"); ap.add_argument("--tag", default="be_lab")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--ksize", type=int, default=9)
    args = ap.parse_args(); set_seed(0)
    inC = n_channels(args.channels); ckpt_out = f"checkpoints/seg_crop_{args.tag}.pt"
    print(f"[cfg] body-edge tag={args.tag} channels={args.channels}({inC}) ksize={args.ksize}", flush=True)
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
    tr = torch.utils.data.DataLoader(PatchSet(items_tr, args.channels, True), batch_size=args.batch,
                                     shuffle=True, num_workers=4, drop_last=True)
    va = torch.utils.data.DataLoader(PatchSet(items_va, args.channels, False), batch_size=args.batch,
                                     shuffle=False, num_workers=2)
    net = BodyEdgeUNet(inC).to(DEVICE)
    print(f"[model] params={sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
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
            x = x.to(DEVICE); t = t.to(DEVICE)
            body_t = erode(t, args.ksize); edge_t = (t - body_t).clamp(0, 1)
            final, b, e = net.forward_train(x)
            loss = dbce(final, t) + dbce(b, body_t) + dbce(e, edge_t)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        sched.step(); vd = val_dice(); star = ""
        if vd > best:
            best = vd; torch.save({"net": net.state_dict(), "arch": "bodyedge",
                                   "channels": args.channels, "val_dice": vd, "epoch": ep}, ckpt_out); star = "  *best"
        print(f"[ep {ep}] loss={tot/max(nb,1):.4f} val_dice={vd:.4f}{star}", flush=True)
    print(f"\nTrain xong. best val={best:.4f} -> {ckpt_out}", flush=True)
    ck = torch.load(ckpt_out, weights_only=False); net.load_state_dict(ck["net"])
    eval_handdraw(net, args.channels, args.tag)

if __name__ == "__main__":
    main()
