"""
HƯỚNG MỚI (user 2026-06-25) — TWO-STAGE CROP + SEGMENTER THUẦN (bỏ box->SAM).
Đánh thẳng nút thắt "SAM tô lố" (spill 0.275): u không tách bạch thị giác khi box lỏng,
SAM khoanh cả tạng. Thay Stage-3 SAM bằng segmenter CHUYÊN DỤNG học trên CROP:

  Stage 1 (Global): detector hiện có (DenseDetector trên SAM-feat) -> box thô. [tái dùng]
  Stage 2 (Local) : crop box + pad 15%, resize 512x512 (model chỉ thấy vùng chật, không bị
                    nền tím / mô lành ở xa làm nhiễu).
  Stage 3 (Seg)   : SegResNet / SwinUNETR thuần trên crop -> mask u trong patch -> ghép union.

Kênh nhân tạo (concat vào RGB):
  - LAB (tách sáng/màu tốt) -> 6ch ; HSV tùy chọn.
  - Texture: Gabor energy (đa hướng) -> +1ch làm nổi kết cấu sần của u.
Loss: dice+bce HOẶC dice+boundary (alpha*Dice + (1-alpha)*Boundary, alpha 1.0->0.7) siết viền.

TRAIN: 318 GT-fragment (connected comp mask THẬT) từ 178 ảnh truth (loại handdraw/e200/t12 patient).
       Box train JITTER (pad 0.1-0.45 + dịch) -> dạy seg bám u trong box LỎNG (không tô cả patch).
TEST : 50 ảnh vẽ tay ĐỘC LẬP. Báo full-auto (box detector) vs 0.666 & ceiling (box GT-mảnh) vs 0.857.

  python seg_crop.py --arch segresnet --channels lab_tex --loss diceboundary --epochs 60 --tag segR_labtex_bd
Env: /home/hvusynh2/conda_envs/medsam2_anno/bin/python (sam2+hydra+monai+skimage).
"""
import argparse, csv, json, os, sys, random, hashlib, math
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)

IMG_DIR = "data/20241212"; TRUTH_DIR = "labels_truth/masks"; HMASK = "labels_handdraw/masks"
SAM_MASK = "labels/masks"
DEVICE = "cuda"; AC = torch.autocast("cuda", dtype=torch.bfloat16)
MIN_FRAC = 0.002; SIZE = 512
CFG = "configs/sam2.1_hiera_t512"; CKPT = "checkpoints/sam2.1_hiera_tiny.pt"; RES = 1024
DET_CKPT = os.environ.get("DET_CKPT", "checkpoints/detector_recall.pt")
DET_THR = float(os.environ.get("DET_THR", "0.5"))

def pat(s): return s.split("^")[0]
def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)

# ---------------- kênh nhân tạo ----------------
_GABOR = None
def gabor_bank():
    """8 kernel Gabor (4 hướng x 2 tần) — dựng 1 lần."""
    global _GABOR
    if _GABOR is None:
        ks = []
        for theta in np.arange(0, np.pi, np.pi / 4):
            for lam in (6.0, 12.0):
                k = cv2.getGaborKernel((15, 15), 3.0, theta, lam, 0.5, 0, ktype=cv2.CV_32F)
                k -= k.mean(); ks.append(k)
        _GABOR = ks
    return _GABOR

def texture_chan(gray):
    """Gabor energy map: max |response| qua bank, chuẩn hóa [0,1]. gray uint8 -> float32 HxW."""
    g = gray.astype(np.float32) / 255.0
    acc = None
    for k in gabor_bank():
        r = np.abs(cv2.filter2D(g, cv2.CV_32F, k))
        acc = r if acc is None else np.maximum(acc, r)
    m = acc.max()
    return acc / m if m > 1e-6 else acc

def make_channels(bgr, mode):
    """bgr crop (HxWx3 uint8) -> (C,H,W) float32 đã chuẩn hóa ~[-1,1] cho RGB/LAB/HSV, [0,1] texture."""
    chans = []
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chans.append((rgb - 0.5) / 0.5)  # (H,W,3)
    if "lab" in mode:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        chans.append((lab - 0.5) / 0.5)
    if "hsv" in mode:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] /= 179.0; hsv[..., 1] /= 255.0; hsv[..., 2] /= 255.0
        chans.append((hsv - 0.5) / 0.5)
    arr = np.concatenate(chans, axis=2)            # (H,W,Crgb)
    out = [arr]
    if "tex" in mode:
        tex = texture_chan(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))[..., None]  # (H,W,1) in [0,1]
        out.append(tex)
    full = np.concatenate(out, axis=2)
    return np.ascontiguousarray(full.transpose(2, 0, 1))  # (C,H,W)

def n_channels(mode):
    c = 3
    if "lab" in mode: c += 3
    if "hsv" in mode: c += 3
    if "tex" in mode: c += 1
    return c

# ---------------- crop ----------------
def pad_box(box, pad, W, H):
    x0, y0, x1, y1 = box; bw = x1 - x0; bh = y1 - y0
    x0 -= bw * pad; x1 += bw * pad; y0 -= bh * pad; y1 += bh * pad
    x0 = max(0, int(round(x0))); y0 = max(0, int(round(y0)))
    x1 = min(W, int(round(x1))); y1 = min(H, int(round(y1)))
    if x1 <= x0: x1 = min(W, x0 + 1)
    if y1 <= y0: y1 = min(H, y0 + 1)
    return [x0, y0, x1, y1]

def frag_boxes(mask):
    m = mask.astype(np.uint8); n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    thr = MIN_FRAC * m.size; out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < thr: continue
        x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
        w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        out.append([float(x), float(y), float(x + w), float(y + h)])
    return out

def ncomp(m):
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return max(1, int((st[1:, cv2.CC_STAT_AREA] >= MIN_FRAC * m.size).sum()))

# ---------------- dataset ----------------
class PatchSet(torch.utils.data.Dataset):
    """Mỗi item = 1 GT-fragment. Train: jitter box (pad+dịch) + aug; Val: pad cố định."""
    def __init__(self, items, mode, train=True):
        self.items = items; self.mode = mode; self.train = train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        stem, box = self.items[i]
        bgr = cv2.imread(f"{IMG_DIR}/{stem}.jpg"); H, W = bgr.shape[:2]
        m = cv2.imread(f"{TRUTH_DIR}/{stem}.png", 0) > 127
        if self.train:
            pad = random.uniform(0.10, 0.45)
            bw = box[2] - box[0]; bh = box[3] - box[1]
            sx = random.uniform(-0.12, 0.12) * bw; sy = random.uniform(-0.12, 0.12) * bh
            b = [box[0] + sx, box[1] + sy, box[2] + sx, box[3] + sy]
        else:
            pad = 0.15; b = box
        x0, y0, x1, y1 = pad_box(b, pad, W, H)
        crop = bgr[y0:y1, x0:x1]; cm = m[y0:y1, x0:x1].astype(np.float32)
        crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        cm = cv2.resize(cm, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        if self.train:
            if random.random() < 0.5: crop = crop[:, ::-1]; cm = cm[:, ::-1]
            if random.random() < 0.5: crop = crop[::-1]; cm = cm[::-1]
            k = random.randint(0, 3)
            if k: crop = np.rot90(crop, k).copy(); cm = np.rot90(cm, k).copy()
            if random.random() < 0.6:  # color jitter nhẹ
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[..., 1] *= random.uniform(0.85, 1.15); hsv[..., 2] *= random.uniform(0.85, 1.15)
                crop = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        x = make_channels(np.ascontiguousarray(crop), self.mode)
        t = (cm > 0.5).astype(np.float32)
        # boundary level-set phi: âm trong u, dương ngoài u
        if t.sum() > 0 and t.sum() < t.size:
            din = ndimage.distance_transform_edt(t)
            dout = ndimage.distance_transform_edt(1 - t)
            phi = (dout - din).astype(np.float32) / SIZE  # chuẩn hóa theo cạnh
        else:
            phi = np.zeros_like(t)
        return torch.from_numpy(x), torch.from_numpy(t)[None], torch.from_numpy(phi)[None]

# ---------------- model ----------------
def build_model(arch, in_ch):
    from monai.networks.nets import SegResNet, SwinUNETR
    if arch == "segresnet":
        return SegResNet(spatial_dims=2, in_channels=in_ch, out_channels=1,
                         init_filters=32, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1))
    if arch == "swinunetr":
        return SwinUNETR(in_channels=in_ch, out_channels=1, spatial_dims=2, feature_size=24)
    raise ValueError(arch)

# ---------------- loss ----------------
def soft_dice(logits, t):
    p = torch.sigmoid(logits.float()); dims = (1, 2, 3)
    inter = (p * t).sum(dims)
    return (1 - (2 * inter + 1) / (p.sum(dims) + t.sum(dims) + 1)).mean()

def loss_fn(logits, t, phi, kind, alpha):
    dice = soft_dice(logits, t)
    bce = F.binary_cross_entropy_with_logits(logits.float(), t.float())
    if kind == "diceboundary":
        p = torch.sigmoid(logits.float())
        boundary = (p * phi).mean()                  # phi âm trong u -> đẩy p cao trong u
        return alpha * (dice + 0.5 * bce) + (1 - alpha) * boundary
    return dice + bce                                 # dicebce

# ---------------- eval full-res trên 50 vẽ tay ----------------
@torch.no_grad()
def seg_union(net, mode, bgr, boxes_px, thr=0.5):
    """Mỗi box -> crop pad0.15 -> seg -> đặt lại full-res -> union."""
    H, W = bgr.shape[:2]; union = np.zeros((H, W), bool)
    for b in boxes_px:
        x0, y0, x1, y1 = pad_box(list(map(float, b)), 0.15, W, H)
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0: continue
        cr = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(make_channels(np.ascontiguousarray(cr), mode))[None].to(DEVICE)
        lg = net(x); pm = torch.sigmoid(lg)[0, 0].float().cpu().numpy()
        pm = cv2.resize(pm, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > thr
        union[y0:y1, x0:x1] |= pm
    return union

def dice_np(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s

def eval_handdraw(net, mode, tag):
    """full-auto (box detector) + ceiling (box GT-mảnh) trên 50 ảnh vẽ tay."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from detector import DenseDetector, propose_boxes
    from specimen_clean import clean_specimen
    net.eval()
    model = build_sam2(CFG, CKPT, device=DEVICE, hydra_overrides_extra=[f"++model.image_size={RES}"])
    predictor = SAM2ImagePredictor(model)
    ck = torch.load(DET_CKPT, weights_only=False)
    det = DenseDetector(grid=ck.get("grid", 64)).to(DEVICE); det.load_state_dict(ck["det"]); det.eval()
    stems = json.load(open("labels_handdraw/select.json"))["stems"]
    have = [s for s in stems if os.path.isfile(f"{HMASK}/{s}.png")]
    rows = []
    for s in have:
        gt = cv2.imread(f"{HMASK}/{s}.png", 0) > 127
        bgr = cv2.imread(f"{IMG_DIR}/{s}.jpg"); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = gt.shape; no = ncomp(gt)
        spec, _, _ = clean_specimen(bgr)
        # Stage1: detector box
        with torch.inference_mode(), AC:
            predictor.set_image(rgb)
            feat = predictor._features["image_embed"].float()
            obj, boxes = det(feat)
        ab = propose_boxes(obj[0].float(), boxes[0].float(), H, W, spec=spec, thr=DET_THR)
        m_auto = seg_union(net, mode, bgr, ab) if len(ab) else np.zeros_like(gt)
        # ceiling: box GT-mảnh từ mask vẽ tay
        gb = frag_boxes(gt)
        m_ceil = seg_union(net, mode, bgr, gb) if gb else np.zeros_like(gt)
        rows.append((no, dice_np(m_auto, gt), dice_np(m_ceil, gt), len(ab)))
        print(f"  {s[:22]:24} n={no} auto={rows[-1][1]:.3f} ceil={rows[-1][2]:.3f} (#box {len(ab)})", flush=True)
    import statistics as st
    def agg(idx, f=lambda r: True):
        v = [r[idx] for r in rows if f(r)]; return (st.median(v), st.mean(v), len(v)) if v else (0, 0, 0)
    one = lambda r: r[0] <= 1; mul = lambda r: r[0] > 1
    A, A1, Am = agg(1), agg(1, one), agg(1, mul)
    C, C1, Cm = agg(2), agg(2, one), agg(2, mul)
    print(f"\n===== SEG-CROP [{tag}] trên {len(rows)} vẽ tay =====")
    print(f"FULL-AUTO (box detector): median={A[0]:.4f} mean={A[1]:.4f} | 1u={A1[0]:.3f}(n{A1[2]}) >1u={Am[0]:.3f}(n{Am[2]})")
    print(f"CEILING   (box GT-mảnh) : median={C[0]:.4f} mean={C[1]:.4f} | 1u={C1[0]:.3f} >1u={Cm[0]:.3f}")
    print(f"SO SÁNH: full-auto box->SAM=0.666 | ceiling box->SAM=0.857 | red-ceiling=0.938 | nhãn SAM cũ=0.554")
    res = {"tag": tag, "full_auto_median": A[0], "full_auto_mean": A[1], "fa_1u": A1[0], "fa_mul": Am[0],
           "ceiling_median": C[0], "ceiling_mean": C[1], "ceil_1u": C1[0], "ceil_mul": Cm[0], "n": len(rows)}
    json.dump(res, open(f"results/seg_crop_{tag}.json", "w"), indent=1)
    with open(f"results/seg_crop_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["n_obj", "dice_auto", "dice_ceiling", "n_box"])
        for r in rows: w.writerow([r[0], round(r[1], 4), round(r[2], 4), r[3]])
    return res

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="segresnet", choices=["segresnet", "swinunetr"])
    ap.add_argument("--channels", default="lab_tex", help="rgb | lab | lab_tex | hsv_tex | lab_hsv_tex")
    ap.add_argument("--loss", default="dicebce", choices=["dicebce", "diceboundary"])
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--n_val", type=int, default=28)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ckpt_out", default=None)
    args = ap.parse_args(); set_seed(0)
    tag = args.tag or f"{args.arch}_{args.channels}_{args.loss}"
    ckpt_out = args.ckpt_out or f"checkpoints/seg_crop_{tag}.pt"
    inC = n_channels(args.channels)
    print(f"[cfg] arch={args.arch} channels={args.channels}({inC}ch) loss={args.loss} tag={tag}", flush=True)

    # split sạch (loại handdraw/e200/t12 patient) -> fragment items
    truth = sorted(f[:-4] for f in os.listdir(TRUTH_DIR) if f.endswith(".png"))
    hd = set(map(pat, json.load(open("labels_handdraw/select.json"))["stems"]))
    e200 = set(pat(r["stem"]) for r in csv.DictReader(open("results/confirm200_ft_vs_zs.csv")))
    t12 = set(map(pat, json.load(open("labels/test_frozen.json"))["test"]))
    excl = hd | e200 | t12
    clean = [s for s in truth if pat(s) not in excl]
    clean.sort(key=lambda s: hashlib.md5(s.encode()).hexdigest())
    val_stems = set(clean[:args.n_val // 2])          # val theo STEM (chống rò rỉ patch cùng ảnh)
    items_tr, items_va = [], []
    for s in clean:
        m = cv2.imread(f"{TRUTH_DIR}/{s}.png", 0) > 127
        for b in frag_boxes(m):
            (items_va if s in val_stems else items_tr).append((s, b))
    print(f"[split] train patch={len(items_tr)} val patch={len(items_va)} "
          f"(từ {len(clean)} ảnh truth, 50 vẽ tay HELD-OUT)", flush=True)

    tr = torch.utils.data.DataLoader(PatchSet(items_tr, args.channels, True),
                                     batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    va = torch.utils.data.DataLoader(PatchSet(items_va, args.channels, False),
                                     batch_size=args.batch, shuffle=False, num_workers=2)
    net = build_model(args.arch, inC).to(DEVICE)
    print(f"[model] params={sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    @torch.no_grad()
    def val_dice(thr=0.5):
        net.eval(); ds = []
        for x, t, _ in va:
            lg = net(x.to(DEVICE)); p = (torch.sigmoid(lg) > thr).cpu().numpy()
            g = t.numpy() > 0.5
            for k in range(len(p)):
                s = p[k].sum() + g[k].sum(); ds.append(1.0 if s == 0 else 2 * (p[k] & g[k]).sum() / s)
        return float(np.mean(ds)) if ds else 0.0

    logf = open(f"results/seg_crop_{tag}_log.csv", "w", newline=""); logw = csv.writer(logf)
    logw.writerow(["epoch", "loss", "alpha", "val_dice"]); logf.flush()
    best = -1.0
    for ep in range(1, args.epochs + 1):
        net.train(); tot = 0.0; nb = 0
        alpha = max(0.7, 1.0 - 0.01 * ep)
        for x, t, phi in tr:
            x = x.to(DEVICE); t = t.to(DEVICE); phi = phi.to(DEVICE)
            lg = net(x); loss = loss_fn(lg, t, phi, args.loss, alpha)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        sched.step()
        vd = val_dice(); star = ""
        if vd > best:
            best = vd; torch.save({"net": net.state_dict(), "arch": args.arch, "channels": args.channels,
                                   "val_dice": vd, "epoch": ep}, ckpt_out); star = "  *best"
        print(f"[ep {ep}] loss={tot/max(nb,1):.4f} alpha={alpha:.2f} val_dice={vd:.4f}{star}", flush=True)
        logw.writerow([ep, round(tot/max(nb,1), 4), round(alpha, 2), round(vd, 4)]); logf.flush()
    logf.close()
    print(f"\nTrain xong. best val(patch)={best:.4f} -> {ckpt_out}", flush=True)

    ck = torch.load(ckpt_out, weights_only=False); net.load_state_dict(ck["net"])
    eval_handdraw(net, args.channels, tag)

if __name__ == "__main__":
    main()
