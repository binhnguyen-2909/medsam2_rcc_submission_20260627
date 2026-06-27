"""
LOCALIZE #D — HỒI QUY THAM SỐ ELLIPSE. Phát hiện: ellipse_of_GT=0.949 > segmenter 0.883
=> shape elip xấp xỉ u rất tốt; vấn đề là dự đoán ĐÚNG tham số (tâm/trục/góc), không phải
fit ngược từ mask spilly. Giải pháp: CNN trên crop -> (cx,cy,ax,ay,sinθ,cosθ); render ELLIPSE
MỀM khả vi -> Dice vs GT crop (train thẳng theo ellipse-Dice). Inference: render ellipse cứng
làm mask cuối. Train 318 GT-fragment (box jitter mô phỏng detector), eval 50 vẽ tay.
  python ellipse_reg.py --epochs 80
So full-auto 0.635 / ceiling 0.883 / ellipse_of_GT 0.949. Env: medsam2_anno.
"""
import argparse, csv, json, os, sys, hashlib, math
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from seg_crop import (PatchSet, n_channels, frag_boxes, set_seed, TRUTH_DIR, pat,
                      make_channels, pad_box, ncomp, SIZE)
DEVICE = "cuda"; RR = 128  # độ phân giải render mềm khi train
CKPT = "checkpoints/ellipse_reg.pt"

class EllipseNet(nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        def cb(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.GroupNorm(8, o), nn.GELU())
        self.f = nn.Sequential(cb(in_ch, 32), cb(32, 64), cb(64, 128), cb(128, 128), cb(128, 256),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.h = nn.Linear(256, 6)
    def forward(self, x):
        o = self.h(self.f(x))
        cx = torch.sigmoid(o[:, 0]); cy = torch.sigmoid(o[:, 1])
        ax = torch.sigmoid(o[:, 2]) * 0.8 + 0.02; ay = torch.sigmoid(o[:, 3]) * 0.8 + 0.02
        ang = o[:, 4:6]; ang = ang / (ang.norm(dim=1, keepdim=True) + 1e-6)  # (sin2θ,cos2θ) chuẩn hóa
        return cx, cy, ax, ay, ang[:, 0], ang[:, 1]

def soft_ellipse(cx, cy, ax, ay, s2, c2, R=RR, k=30.0):
    """render mask mềm (B,1,R,R) trong [0,1]^2 từ tham số. s2,c2 = sin2θ,cos2θ -> θ."""
    B = cx.shape[0]; dev = cx.device
    th = 0.5 * torch.atan2(s2, c2); cs = torch.cos(th); sn = torch.sin(th)
    ys = torch.linspace(0, 1, R, device=dev); xs = torch.linspace(0, 1, R, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")          # (R,R)
    gx = gx[None]; gy = gy[None]
    dx = gx - cx[:, None, None]; dy = gy - cy[:, None, None]
    u = dx * cs[:, None, None] + dy * sn[:, None, None]
    v = -dx * sn[:, None, None] + dy * cs[:, None, None]
    q = (u / ax[:, None, None]) ** 2 + (v / ay[:, None, None]) ** 2
    return torch.sigmoid(k * (1 - q))[:, None]              # (B,1,R,R)

def dice_bce(p, t):
    dims = (1, 2, 3); inter = (p * t).sum(dims)
    d = 1 - (2 * inter + 1) / (p.sum(dims) + t.sum(dims) + 1)
    bce = F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), t, reduction="none").mean(dims)
    return (d + bce).mean()

def load_net(in_ch):
    net = EllipseNet(in_ch).to(DEVICE); net.load_state_dict(torch.load(CKPT, weights_only=False)["net"]); net.eval()
    return net

# ---- render ellipse CỨNG (cv2) ở 512 rồi đặt lại ----
@torch.no_grad()
def ellipse_mask_512(params):
    cx, cy, ax, ay, s2, c2 = params
    th = 0.5 * math.atan2(s2, c2); deg = math.degrees(th)
    m = np.zeros((SIZE, SIZE), np.uint8)
    cv2.ellipse(m, (int(cx * SIZE), int(cy * SIZE)),
                (max(1, int(ax * SIZE)), max(1, int(ay * SIZE))), deg, 0, 360, 1, -1)
    return m

@torch.no_grad()
def predict_union(net, mode, bgr, boxes):
    H, W = bgr.shape[:2]; union = np.zeros((H, W), bool)
    for b in boxes:
        x0, y0, x1, y1 = pad_box(list(map(float, b)), 0.15, W, H)
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0: continue
        cr = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(make_channels(np.ascontiguousarray(cr), mode))[None].to(DEVICE)
        cx, cy, ax, ay, s2, c2 = [v.item() for v in net(x)]
        em = ellipse_mask_512((cx, cy, ax, ay, s2, c2))
        em = cv2.resize(em, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
        union[y0:y1, x0:x1] |= em
    return union

def dice_np(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s

def eval_handdraw(net, mode):
    from detector import DenseDetector, propose_boxes
    from specimen_clean import clean_specimen
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import statistics as st
    AC = torch.autocast("cuda", dtype=torch.bfloat16)
    model = build_sam2("configs/sam2.1_hiera_t512", "checkpoints/sam2.1_hiera_tiny.pt",
                       device=DEVICE, hydra_overrides_extra=["++model.image_size=1024"])
    pred = SAM2ImagePredictor(model)
    dk = torch.load("checkpoints/detector_recall.pt", weights_only=False)
    det = DenseDetector(grid=dk.get("grid", 64)).to(DEVICE); det.load_state_dict(dk["det"]); det.eval()
    stems = json.load(open("labels_handdraw/select.json"))["stems"]
    have = [s for s in stems if os.path.isfile(f"labels_handdraw/masks/{s}.png")]
    rows = []
    for s in have:
        gt = cv2.imread(f"labels_handdraw/masks/{s}.png", 0) > 127
        bgr = cv2.imread(f"data/20241212/{s}.jpg"); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = gt.shape; spec, _, _ = clean_specimen(bgr)
        with torch.inference_mode(), AC:
            pred.set_image(rgb); feat = pred._features["image_embed"].float(); obj, bx = det(feat)
        ab = list(propose_boxes(obj[0].float(), bx[0].float(), H, W, spec=spec, thr=0.5))
        gb = frag_boxes(gt)
        da = dice_np(predict_union(net, mode, bgr, ab) if ab else np.zeros_like(gt), gt)
        dc = dice_np(predict_union(net, mode, bgr, gb) if gb else np.zeros_like(gt), gt)
        rows.append((ncomp(gt), da, dc))
        print(f"  {s[:20]:22} auto={da:.3f} ceil={dc:.3f}", flush=True)
    def agg(i, f=lambda r: True):
        v = [r[i] for r in rows if f(r)]; return st.median(v), st.mean(v)
    one = lambda r: r[0] <= 1; mul = lambda r: r[0] > 1
    A, A1, Am = agg(1), agg(1, one), agg(1, mul); C, C1, Cm = agg(2), agg(2, one), agg(2, mul)
    print(f"\n===== ELLIPSE-REG trên 50 vẽ tay =====")
    print(f"FULL-AUTO (box detector): median={A[0]:.4f} mean={A[1]:.4f} | 1u={A1[0]:.3f} >1u={Am[0]:.3f}")
    print(f"CEILING   (box GT-mảnh) : median={C[0]:.4f} mean={C[1]:.4f} | 1u={C1[0]:.3f} >1u={Cm[0]:.3f}")
    print(f"SO: full-auto 0.635 | ceiling segmenter 0.883 | ellipse_of_GT 0.949")
    json.dump({"full_auto_median": A[0], "full_auto_mean": A[1], "fa_1u": A1[0], "fa_mul": Am[0],
               "ceiling_median": C[0], "ceiling_mean": C[1], "ceil_1u": C1[0], "ceil_mul": Cm[0]},
              open("results/loc_ellipse_reg.json", "w"), indent=1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--channels", default="lab")
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args(); set_seed(0); inC = n_channels(args.channels)
    truth = sorted(f[:-4] for f in os.listdir(TRUTH_DIR) if f.endswith(".png"))
    hd = set(map(pat, json.load(open("labels_handdraw/select.json"))["stems"]))
    e200 = set(pat(r["stem"]) for r in csv.DictReader(open("results/confirm200_ft_vs_zs.csv")))
    t12 = set(map(pat, json.load(open("labels/test_frozen.json"))["test"]))
    clean = [s for s in truth if pat(s) not in (hd | e200 | t12)]
    clean.sort(key=lambda s: hashlib.md5(s.encode()).hexdigest())
    val_stems = set(clean[:14]); items_tr, items_va = [], []
    for s in clean:
        m = cv2.imread(f"{TRUTH_DIR}/{s}.png", 0) > 127
        for b in frag_boxes(m):
            (items_va if s in val_stems else items_tr).append((s, b))
    print(f"[ellipse-reg] train={len(items_tr)} val={len(items_va)} frag ({args.channels},{inC}ch)", flush=True)
    tr = torch.utils.data.DataLoader(PatchSet(items_tr, args.channels, True), batch_size=args.batch,
                                     shuffle=True, num_workers=4, drop_last=True)
    va = torch.utils.data.DataLoader(PatchSet(items_va, args.channels, False), batch_size=args.batch, num_workers=2)
    net = EllipseNet(inC).to(DEVICE); opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    @torch.no_grad()
    def val_dice():
        net.eval(); ds = []
        for x, t, _ in va:
            p = net(x.to(DEVICE)); m = soft_ellipse(*p)
            td = F.interpolate(t.to(DEVICE), size=(RR, RR), mode="area")
            pm = (m > 0.5).float(); gg = (td > 0.5).float(); dims = (1, 2, 3)
            inter = (pm * gg).sum(dims); s = pm.sum(dims) + gg.sum(dims)
            ds += [(1.0 if s[i] == 0 else (2 * inter[i] / s[i]).item()) for i in range(len(s))]
        return float(np.mean(ds)) if ds else 0.0
    best = -1
    for ep in range(1, args.epochs + 1):
        net.train(); tot = 0; nb = 0
        for x, t, _ in tr:
            x = x.to(DEVICE); td = F.interpolate(t.to(DEVICE), size=(RR, RR), mode="area")
            p = net(x); m = soft_ellipse(*p); loss = dice_bce(m, (td > 0.5).float())
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        sched.step(); vd = val_dice(); star = ""
        if vd > best: best = vd; torch.save({"net": net.state_dict(), "channels": args.channels, "epoch": ep, "val": vd}, CKPT); star = "  *best"
        if ep % 5 == 0 or star: print(f"[ep {ep}] loss={tot/max(nb,1):.4f} val_ellDice={vd:.4f}{star}", flush=True)
    print(f"\nTrain xong. best val={best:.4f} -> {CKPT}", flush=True)
    net.load_state_dict(torch.load(CKPT, weights_only=False)["net"])
    eval_handdraw(net, args.channels)

if __name__ == "__main__":
    main()
