"""
FINE-TUNE "CÁI ĐỎ" (SAM2.1 prompt_encoder + mask_decoder) trên MASK THẬT (labels_truth)
— mục tiêu: SAM thôi TÔ LỐ ra ngoài u khi box hơi lỏng (spill 0.27 trên handdraw).
Chẩn đoán red-ceiling: nếu SAM khoanh hoàn hảo trong box detector hiện tại -> full-auto 0.666->0.938.

Khác finetune_sam2.py: (1) target = mask THẬT (labels_truth ưu tiên, else SAM cũ);
(2) jitter box RỘNG hơn (frac 0.25) để dạy decoder bám u, không phình theo box lỏng;
(3) split nội bộ từ stems CÓ mask thật, giữ 12 làm val chọn epoch; 50 ảnh vẽ tay KHÔNG đụng (held-out).
Image encoder ĐÓNG BĂNG (đặc trưng precompute 1 lần, cache CPU). Eval cuối = eval_red_handdraw.py.

  python finetune_red_truth.py --epochs 40 --batch 12 --jitter 0.25
Lưu best theo val Dice -> checkpoints/sam2.1_rcc_red_truth.pt
"""
import argparse, csv, json, os, random, sys, hashlib
import cv2, numpy as np, torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

IMG_DIR = os.path.join(ROOT, "data/20241212")
TRUTH_DIR = os.path.join(ROOT, "labels_truth/masks")
SAM_DIR = os.path.join(ROOT, "labels/masks")
CONFIG = os.environ.get("SAM2_CONFIG", "configs/sam2.1_hiera_t512")
CKPT = os.environ.get("SAM2_CKPT", "checkpoints/sam2.1_hiera_tiny.pt")
RES = int(os.environ.get("SAM2_RES", "1024"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AC = torch.autocast("cuda", dtype=torch.bfloat16)
MIN_FRAC = 0.002
def pat(s): return s.split("^")[0]

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def read_rgb(stem): return cv2.cvtColor(cv2.imread(os.path.join(IMG_DIR, stem + ".jpg")), cv2.COLOR_BGR2RGB)
def mask_path(stem):
    tp = os.path.join(TRUTH_DIR, stem + ".png")
    return tp if os.path.isfile(tp) else os.path.join(SAM_DIR, stem + ".png")
def read_mask(stem):
    m = cv2.imread(mask_path(stem), cv2.IMREAD_GRAYSCALE); return (m > 127).astype(np.uint8)
def components(mask):
    n, lab = cv2.connectedComponents(mask, connectivity=8); out = []; thr = MIN_FRAC * mask.size
    for i in range(1, n):
        c = (lab == i)
        if c.sum() >= thr: out.append(c)
    if not out and mask.sum() > 0: out = [mask.astype(bool)]
    return out
def bbox_of(m): ys, xs = np.where(m); return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
def jitter_box(box, W, H, frac):
    x0, y0, x1, y1 = box; bw, bh = x1 - x0, y1 - y0
    dx = random.uniform(-frac, frac) * bw; dy = random.uniform(-frac, frac) * bh
    ex = random.uniform(0, frac) * bw; ey = random.uniform(0, frac) * bh   # nới rộng -> mô phỏng box lỏng của detector
    return [max(0, x0 + dx - ex), max(0, y0 + dy - ey), min(W - 1, x1 + dx + ex), min(H - 1, y1 + dy + ey)]

def decode_batch(model, image_embed, high_res_feats, orig_hw, boxes, transforms):
    b = torch.as_tensor(boxes, dtype=torch.float, device=DEVICE)
    bt = transforms.transform_boxes(b, normalize=True, orig_hw=orig_hw)
    box_coords = bt.reshape(-1, 2, 2); B = box_coords.shape[0]
    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=DEVICE).repeat(B, 1)
    sparse, dense = model.sam_prompt_encoder(points=(box_coords, box_labels), boxes=None, masks=None)
    low_res, iou, _, _ = model.sam_mask_decoder(
        image_embeddings=image_embed, image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False, repeat_image=False, high_res_features=high_res_feats)
    return low_res
def dice_bce_batch(logits, target):
    p = torch.sigmoid(logits.float()); t = target.float(); dims = (1, 2, 3)
    inter = (p * t).sum(dims)
    dice = 1 - (2 * inter + 1) / (p.sum(dims) + t.sum(dims) + 1)
    bce = F.binary_cross_entropy_with_logits(logits.float(), t, reduction="none").mean(dims)
    return (dice + bce).mean()

class _DS(torch.utils.data.Dataset):
    def __init__(self, stems): self.stems = stems
    def __len__(self): return len(self.stems)
    def __getitem__(self, i):
        s = self.stems[i]; gt = read_mask(s); H, W = gt.shape
        return s, read_rgb(s), components(gt), (W, H)

@torch.no_grad()
def eval_dice(predictor, stems):
    """Dice per-fragment với bbox SÁT (không jitter) — trần box->mask trên mask thật."""
    predictor.model.eval(); ds = []
    for s in stems:
        rgb = read_rgb(s); gt = read_mask(s)
        with AC: predictor.set_image(rgb)
        for c in components(gt):
            box = np.array(bbox_of(c), dtype=np.float32)
            with AC: masks, sc, _ = predictor.predict(box=box, multimask_output=False)
            pred = masks[0].astype(bool); inter = (pred & c).sum()
            ds.append(float(2 * inter / (pred.sum() + c.sum() + 1e-9)))
    return float(np.mean(ds)) if ds else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--precompute_batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--jitter", type=float, default=0.25)
    ap.add_argument("--n_val", type=int, default=12)
    ap.add_argument("--ckpt_out", default=os.path.join(ROOT, "checkpoints/sam2.1_rcc_red_truth.pt"))
    args = ap.parse_args(); set_seed(0)

    # split: stems CÓ mask thật, loại bệnh nhân handdraw/eval200/test12, giữ n_val làm val
    truth = sorted(f[:-4] for f in os.listdir(TRUTH_DIR) if f.endswith(".png"))
    hd = set(map(pat, json.load(open("labels_handdraw/select.json"))["stems"]))
    e200 = set(pat(r["stem"]) for r in csv.DictReader(open("results/confirm200_ft_vs_zs.csv")))
    t12 = set(map(pat, json.load(open("labels/test_frozen.json"))["test"]))
    excl = hd | e200 | t12
    clean = [s for s in truth if pat(s) not in excl]
    clean.sort(key=lambda s: hashlib.md5(s.encode()).hexdigest())
    val = clean[:args.n_val]; train = clean[args.n_val:]
    json.dump({"train": train, "val": val}, open("labels/split_red_truth.json", "w"))
    print(f"[split] truth train={len(train)} val={len(val)} (handdraw 50 HELD-OUT, 0 rò rỉ) | jitter={args.jitter}", flush=True)

    print(f"Nạp SAM2.1@{RES} ...", flush=True)
    model = build_sam2(CONFIG, CKPT, device=DEVICE, hydra_overrides_extra=[f"++model.image_size={RES}"])
    predictor = SAM2ImagePredictor(model); transforms = predictor._transforms
    for p in model.parameters(): p.requires_grad_(False)
    params = []
    for m in [model.sam_prompt_encoder, model.sam_mask_decoder]:
        for p in m.parameters(): p.requires_grad_(True); params.append(p)
    print(f"Tham số train (đỏ): {sum(p.numel() for p in params)/1e6:.2f}M", flush=True)

    loader = torch.utils.data.DataLoader(_DS(train), batch_size=args.precompute_batch,
        num_workers=args.workers, collate_fn=lambda b: b, persistent_workers=False)
    def chunk(items):
        try:
            with AC: predictor.set_image_batch([it[1] for it in items])
            ie = predictor._features["image_embed"]; hrf = predictor._features["high_res_feats"]; out = []
            for i, (s, rgb, comps, (W, H)) in enumerate(items):
                feats = {"image_embed": ie[i:i+1].detach().to("cpu"),
                         "high_res_feats": [f[i:i+1].detach().to("cpu") for f in hrf]}
                ohw = tuple(predictor._orig_hw[i])
                for c in comps: out.append({"feats": feats, "orig_hw": ohw, "comp": c, "WH": (W, H)})
            return out
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(items) == 1: raise
            mid = len(items)//2; return chunk(items[:mid]) + chunk(items[mid:])
    cache = []
    for batch in loader: cache.extend(chunk(batch))
    print(f"Mẫu train (cụm u): {len(cache)}", flush=True)

    from collections import defaultdict
    groups = defaultdict(list)
    for i, it in enumerate(cache): groups[it["orig_hw"]].append(i)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    logf = open("results/finetune_red_truth_log.csv", "w", newline=""); logw = csv.writer(logf)
    logw.writerow(["epoch", "train_loss", "val_dice"]); logf.flush()

    base = eval_dice(predictor, val)
    print(f"[ep 0] zero-shot val Dice (trần box->mask, mask thật) = {base:.4f}", flush=True)
    logw.writerow([0, "", round(base, 4)]); logf.flush()
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train(); order = []
        for k, idxs in groups.items():
            idxs = idxs[:]; random.shuffle(idxs)
            for i in range(0, len(idxs), args.batch): order.append(idxs[i:i+args.batch])
        random.shuffle(order); tot = 0.0; nb = 0
        for bidx in order:
            items = [cache[i] for i in bidx]
            image_embed = torch.stack([it["feats"]["image_embed"][0] for it in items]).to(DEVICE)
            nlv = len(items[0]["feats"]["high_res_feats"])
            hrf = [torch.stack([it["feats"]["high_res_feats"][l][0] for it in items]).to(DEVICE) for l in range(nlv)]
            ohw = items[0]["orig_hw"]
            boxes = [jitter_box(bbox_of(it["comp"]), *it["WH"], args.jitter) for it in items]
            tg = [(cv2.resize(it["comp"].astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA) > 0.5).astype(np.float32) for it in items]
            target = torch.as_tensor(np.stack(tg), device=DEVICE)[:, None]
            with AC: logits = decode_batch(model, image_embed, hrf, ohw, boxes, transforms)
            loss = dice_bce_batch(logits, target)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        tl = tot / max(nb, 1); vd = eval_dice(predictor, val)
        star = ""
        if vd > best:
            best = vd
            torch.save({"model": model.state_dict(), "val_dice": vd, "epoch": ep,
                        "zero_shot_val": base, "jitter": args.jitter}, args.ckpt_out); star = "  *best->lưu"
        print(f"[ep {ep}] train_loss={tl:.4f} | val Dice={vd:.4f}{star}", flush=True)
        logw.writerow([ep, round(tl, 4), round(vd, 4)]); logf.flush()
    logf.close()
    print(f"\nXong. zero-shot val={base:.4f} | BEST FT val={best:.4f} -> {args.ckpt_out}", flush=True)

if __name__ == "__main__":
    main()
