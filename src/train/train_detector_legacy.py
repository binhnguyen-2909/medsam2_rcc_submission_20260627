"""
Train DETR-style multi-box detector trên SAM2 (encoder đông cứng).
  Stage A  Hungarian match box đề xuất <-> box GT (cls focal + L1 + GIoU)
  Stage B  box đã match -> SAM prompt+mask decoder (đông cứng, KHẢ VI theo box)
           -> mask trong box vs mask tay (Dice+BCE), gradient chỉnh cả box.

Không rò rỉ: loại bệnh nhân eval200 (results/confirm200) + test_frozen khỏi train.
Eval end-to-end (auto multi-box -> SAM -> Dice vs mask tay) chạy ở eval_detector_e2e.py.

  python train_detector.py --epochs 80 --batch 8
Lưu best (val end-to-end Dice) -> checkpoints/detector.pt
"""
import argparse, csv, json, os, sys, hashlib, random, time
import numpy as np, cv2, torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from detector import (DenseDetector, assign_targets, dense_losses,
                      decode_detections, cxcywh_to_xyxy)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

IMG_DIR = "data/20241212"; MASK_DIR = "labels/masks"; TRUTH_DIR = "labels_truth/masks"
def mask_path(stem):
    """Ưu tiên mask THẬT (labels_truth, vẽ tay đầy đủ mảnh) nếu có; else mask SAM cũ."""
    tp = f"{TRUTH_DIR}/{stem}.png"
    return tp if os.path.isfile(tp) else f"{MASK_DIR}/{stem}.png"
CFG = "configs/sam2.1_hiera_t512"; CKPT = "checkpoints/sam2.1_hiera_tiny.pt"
RES = 1024; DEVICE = "cuda"
AC = torch.autocast("cuda", dtype=torch.bfloat16)
CACHE = "cache_feats"; os.makedirs(CACHE, exist_ok=True)
MIN_FRAC = 0.001                      # cụm < 0.1% diện tích ảnh -> bỏ
def pat(s): return s.split("^")[0]


# ---------------- GT: cụm mask -> box cxcywh norm + mask 256 ----------------
def gt_targets(stem):
    m = cv2.imread(mask_path(stem), 0)
    if m is None: return None
    m = (m > 127).astype(np.uint8); H, W = m.shape
    n, lab = cv2.connectedComponents(m, connectivity=8)
    boxes, comps = [], []
    thr = MIN_FRAC * m.size
    for i in range(1, n):
        c = (lab == i)
        if c.sum() < thr: continue
        ys, xs = np.where(c)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        cx = (x0 + x1) / 2 / W; cy = (y0 + y1) / 2 / H
        bw = (x1 - x0 + 1) / W; bh = (y1 - y0 + 1) / H
        boxes.append([cx, cy, bw, bh])
        comps.append(cv2.resize(c.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA))
    if not boxes: return None
    return (np.array(boxes, np.float32), (np.stack(comps) > 0.5).astype(np.uint8), (H, W))


# ---------------- precompute encoder feats -> đĩa ----------------
def cache_path(stem, flip): return f"{CACHE}/{stem}{'__f' if flip else ''}.pt"

def precompute(predictor, stems, flips, log_every=50):
    def stale(s, fl):
        cp = cache_path(s, fl)
        if not os.path.isfile(cp): return True
        mp = mask_path(s)   # mask thật mới vẽ -> cache (box GT cũ) lỗi thời, rebuild
        return os.path.isfile(mp) and os.path.getmtime(mp) > os.path.getmtime(cp)
    todo = [(s, fl) for s in stems for fl in flips if stale(s, fl)]
    print(f"[precompute] cần {len(todo)} (đã có {len(stems)*len(flips)-len(todo)})", flush=True)
    t0 = time.time()
    for k, (s, fl) in enumerate(todo):
        bgr = cv2.imread(f"{IMG_DIR}/{s}.jpg")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt = gt_targets(s)
        if gt is None: continue
        boxes, comps, hw = gt
        if fl:
            rgb = rgb[:, ::-1].copy()
            comps = comps[:, :, ::-1].copy()
            boxes = boxes.copy(); boxes[:, 0] = 1.0 - boxes[:, 0]   # flip cx
        with torch.inference_mode(), AC:
            predictor.set_image(rgb)
        feat = predictor._features["image_embed"][0].half().cpu()
        hrf = [f[0].half().cpu() for f in predictor._features["high_res_feats"]]
        torch.save({"feat": feat, "hrf": hrf, "boxes": boxes, "comps": comps, "hw": hw},
                   cache_path(s, fl))
        if (k + 1) % log_every == 0:
            print(f"  {k+1}/{len(todo)}  ({(time.time()-t0)/(k+1):.2f}s/ảnh)", flush=True)


# ---------------- SAM decode (khả vi theo box) ----------------
def decode_masks(model, transforms, feat, hrf, hw, boxes_cxcywh, out256=True):
    """feat(256,64,64) hrf list, boxes(n,4) cxcywh norm orig -> mask logits.
    out256 -> (n,256,256) cho train loss; else upsample (n,H,W) cho eval."""
    H, W = hw
    xyxy = cxcywh_to_xyxy(boxes_cxcywh).clamp(0, 1)
    scale = torch.tensor([W, H, W, H], device=boxes_cxcywh.device, dtype=boxes_cxcywh.dtype)
    abs_box = xyxy * scale
    bt = transforms.transform_boxes(abs_box, normalize=True, orig_hw=hw)   # (n,2,2)
    box_coords = bt.reshape(-1, 2, 2)
    n = box_coords.shape[0]
    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=DEVICE).repeat(n, 1)
    sparse, dense = model.sam_prompt_encoder(points=(box_coords, box_labels), boxes=None, masks=None)
    low_res, iou, _, _ = model.sam_mask_decoder(
        image_embeddings=feat[None], image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False, repeat_image=True, high_res_features=[h[None] for h in hrf])
    if out256:
        return F.interpolate(low_res, (256, 256), mode="bilinear", align_corners=False)[:, 0]
    return F.interpolate(low_res, (H, W), mode="bilinear", align_corners=False)[:, 0]


def dice_bce_loss(logits, target):
    p = logits.sigmoid(); t = target.float()
    inter = (p * t).sum((-1, -2))
    dice = 1 - (2 * inter + 1) / (p.sum((-1, -2)) + t.sum((-1, -2)) + 1)
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none").mean((-1, -2))
    return (dice + bce).mean()


# ---------------- val end-to-end Dice (chọn epoch) — 256px, chống OOM ----------------
@torch.no_grad()
def val_dice(det, model, transforms, stems, thrs=(0.2, 0.3, 0.5, 0.7)):
    """Quét ngưỡng objectness, trả (best_median_dice, best_thr). Tính ở 256px (rẻ, không
    OOM trên GPU chia sẻ). objectness>thr -> NMS -> box -> SAM decode -> union."""
    det.eval()
    per_thr = {t: [] for t in thrs}
    for s in stems:
        ck = torch.load(cache_path(s, False), weights_only=False)
        feat = ck["feat"].float().to(DEVICE); hrf = [h.float().to(DEVICE) for h in ck["hrf"]]
        hw = ck["hw"]
        gt = cv2.imread(mask_path(s), 0) > 127
        gt256 = cv2.resize(gt.astype(np.uint8), (256, 256), interpolation=cv2.INTER_AREA) > 0
        try:
            with AC:
                obj, boxes = det(feat[None])
            for t in thrs:
                bxk, _ = decode_detections(obj[0].float(), boxes[0].float(), thr=t)
                with AC:
                    mk = decode_masks(model, transforms, feat, hrf, hw, bxk, out256=True)
                pred = (mk.sigmoid() > 0.5).any(0).cpu().numpy()
                s_ = pred.sum() + gt256.sum()
                per_thr[t].append(1.0 if s_ == 0 else 2 * (pred & gt256).sum() / s_)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
    meds = {t: float(np.median(v)) if v else 0.0 for t, v in per_thr.items()}
    best_t = max(meds, key=meds.get)
    return meds[best_t], best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--mask_warmup", type=int, default=6)
    ap.add_argument("--mask_w", type=float, default=1.0)
    ap.add_argument("--min_pos", type=int, default=3,
                    help="top-k ô gần tâm/GT ép dương (center-sampling, tăng recall mảnh nhỏ)")
    ap.add_argument("--grid", type=int, default=64,
                    help="độ phân giải lưới detector (64=stride16; 128=stride8 cho mảnh nhỏ)")
    ap.add_argument("--eval_every", type=int, default=4)
    ap.add_argument("--no_flip", action="store_true")
    ap.add_argument("--ckpt_out", default="checkpoints/detector.pt")
    args = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    # ---- split không rò rỉ ----
    done = [s for s in json.load(open("labels/done.json")) if os.path.isfile(f"{MASK_DIR}/{s}.png")]
    eval200 = [r["stem"] for r in csv.DictReader(open("results/confirm200_ft_vs_zs.csv"))]
    test12 = json.load(open("labels/test_frozen.json"))["test"]
    excl = set(map(pat, eval200)) | set(map(pat, test12))
    pool = [s for s in done if pat(s) not in excl]
    pool_pat = sorted(set(map(pat, pool)), key=lambda p: hashlib.md5(p.encode()).hexdigest())
    # CHỌN EPOCH theo Dice-vs-nhãn-THẬT (loại confound cũ: val trước đây là 10 bệnh nhân đầu
    # theo hash, mask chủ yếu là nhãn SAM đếm-sót-mảnh, nên best-by-val bị kéo về hành vi miss
    # của SAM). Reserve ~15 STEM có mask thật làm val_eval; cả bệnh nhân của chúng bị loại khỏi
    # train (chống rò rỉ). val_dice chỉ chấm trên val_eval (toàn nhãn thật). Fallback hash nếu chưa đủ.
    truth_stems = sorted([s for s in pool if os.path.isfile(f"{TRUTH_DIR}/{s}.png")],
                         key=lambda s: hashlib.md5(s.encode()).hexdigest())
    if len(truth_stems) >= 15:
        val_eval = truth_stems[:15]
        val_pat = set(map(pat, val_eval))
    else:
        val_pat = set(pool_pat[:10])
        val_eval = None
    val = sorted(s for s in pool if pat(s) in val_pat)
    train = sorted(s for s in pool if pat(s) not in val_pat)
    if val_eval is None:
        val_eval = val
    n_val_truth = sum(1 for s in val_eval if os.path.isfile(f"{TRUTH_DIR}/{s}.png"))
    n_train_truth = sum(1 for s in train if os.path.isfile(f"{TRUTH_DIR}/{s}.png"))
    json.dump({"train": train, "val": val, "val_eval": val_eval, "eval200": eval200, "test12": test12},
              open("labels/split_detector.json", "w"))
    print(f"[split] train={len(train)} val={len(val)} val_eval={len(val_eval)} (eval200={len(eval200)} test12={len(test12)} loại khỏi train)", flush=True)
    print(f"[split] mask THẬT: val_eval={n_val_truth}/{len(val_eval)} (chọn epoch theo nhãn thật) | train={n_train_truth}", flush=True)

    # ---- SAM (đông cứng) ----
    print("Nạp SAM2.1 tiny@1024 (đông cứng) ...", flush=True)
    model = build_sam2(CFG, CKPT, device=DEVICE, hydra_overrides_extra=[f"++model.image_size={RES}"])
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    predictor = SAM2ImagePredictor(model); transforms = predictor._transforms

    # ---- precompute feats ----
    flips = [False] if args.no_flip else [False, True]
    precompute(predictor, train, flips)
    precompute(predictor, val, [False])

    # danh sách mẫu train (stem, flip) có cache hợp lệ
    samples = [(s, fl) for s in train for fl in flips if os.path.isfile(cache_path(s, fl))]
    print(f"Mẫu train (kể cả flip): {len(samples)}", flush=True)

    det = DenseDetector(grid=args.grid).to(DEVICE)
    opt = torch.optim.AdamW(det.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f"Tham số detector: {sum(p.numel() for p in det.parameters())/1e6:.2f}M", flush=True)

    logf = open("results/detector_log.csv", "w", newline=""); logw = csv.writer(logf)
    logw.writerow(["epoch", "loss", "cls", "l1", "giou", "mask", "val_dice"]); logf.flush()
    best = -1.0
    for ep in range(1, args.epochs + 1):
        det.train(); random.shuffle(samples)
        use_mask = ep >= args.mask_warmup
        agg = {"loss": 0, "cls": 0, "l1": 0, "giou": 0, "mask": 0}; nb = 0; n_oom = 0
        for i in range(0, len(samples), args.batch):
            bs = samples[i:i + args.batch]
            feats, cks = [], []
            for s, fl in bs:
                ck = torch.load(cache_path(s, fl), weights_only=False)
                feats.append(ck["feat"].float()); cks.append(ck)
            try:
                feat = torch.stack(feats).to(DEVICE)               # (B,256,64,64)
                with AC:
                    obj, boxes = det(feat)                          # (B,G,G),(B,G,G,4)
                obj = obj.float(); boxes = boxes.float()
                loss = 0.0; parts = {"cls": 0, "l1": 0, "giou": 0, "mask": 0}
                for bi, ck in enumerate(cks):
                    gtb = torch.as_tensor(ck["boxes"], device=DEVICE)
                    pos, tgt_box, gidx = assign_targets(det.centers, gtb, min_pos_per_gt=args.min_pos)
                    dl = dense_losses(obj[bi], boxes[bi], pos, tgt_box)
                    l = dl["cls"] + 5 * dl["l1"] + 2 * dl["giou"]
                    parts["cls"] += float(dl["cls"]); parts["l1"] += float(dl["l1"]); parts["giou"] += float(dl["giou"])
                    if use_mask and pos.any():
                        # 1 ô đại diện / GT (gần tâm nhất) -> decode mask vs cụm tương ứng
                        gi_pos = gidx[pos]; pb = boxes[bi][pos]
                        sel, segb = [], []
                        for g in gi_pos.unique():
                            idxs = (gi_pos == g).nonzero().squeeze(1)
                            sel.append(pb[idxs[len(idxs) // 2]]); segb.append(int(g))
                        pbs = torch.stack(sel)
                        hrf = [h.float().to(DEVICE) for h in ck["hrf"]]
                        comps = torch.as_tensor(ck["comps"][np.array(segb)], device=DEVICE).float()
                        with AC:
                            ml = decode_masks(model, transforms, feat[bi], hrf, ck["hw"], pbs)
                        mloss = dice_bce_loss(ml.float(), comps)
                        l = l + args.mask_w * mloss; parts["mask"] += float(mloss)
                    loss = loss + l
                loss = loss / len(cks)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(det.parameters(), 1.0); opt.step()
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); n_oom += 1; continue
            agg["loss"] += float(loss); nb += 1
            for k in parts: agg[k] += parts[k] / len(cks)
        for k in agg: agg[k] /= max(1, nb)
        n_steps = (len(samples) + args.batch - 1) // args.batch
        # CHẶN KẾT QUẢ GIẢ: nếu phần lớn batch bị OOM-skip thì epoch KHÔNG học -> không eval/lưu
        if nb < 0.5 * n_steps:
            print(f"[ep {ep}] ⚠️ BỎ QUA: {n_oom}/{n_steps} batch OOM (GPU thiếu RAM) — epoch không học, "
                  f"không eval/lưu. Giảm --batch hoặc chờ GPU trống.", flush=True)
            logw.writerow([ep, "OOM", "", "", "", "", ""]); logf.flush(); continue
        vd = ""; vt = ""
        if ep % args.eval_every == 0 or ep == args.epochs:
            vd, vt = val_dice(det, model, transforms, val_eval)
            # chỉ lưu best SAU khi mask-loss bật (tránh chọn nhầm epoch box-only sớm do val nhiễu);
            # NHƯNG nếu chạy BOX-ONLY (mask_warmup>epochs) thì lưu best box-only bình thường.
            box_only = args.mask_warmup > args.epochs
            if vd > best and (ep >= args.mask_warmup or box_only):
                best = vd
                torch.save({"det": det.state_dict(), "grid": args.grid,
                            "val_dice": vd, "obj_thr": vt, "epoch": ep}, args.ckpt_out)
        print(f"[ep {ep}] loss={agg['loss']:.3f} cls={agg['cls']:.3f} l1={agg['l1']:.3f} "
              f"giou={agg['giou']:.3f} mask={agg['mask']:.3f}"
              + (f" | val_dice={vd:.4f}@thr{vt}{'  *best' if vd==best and vd!='' else ''}" if vd != "" else ""), flush=True)
        logw.writerow([ep, round(agg["loss"], 4), round(agg["cls"], 4), round(agg["l1"], 4),
                       round(agg["giou"], 4), round(agg["mask"], 4), round(vd, 4) if vd != "" else ""])
        logf.flush()
    logf.close()
    print(f"\nXong. best val end-to-end Dice = {best:.4f} -> {args.ckpt_out}", flush=True)


if __name__ == "__main__":
    main()
