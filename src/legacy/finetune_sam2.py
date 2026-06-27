"""
Fine-tune SAM2.1 (box->mask) cho KHỐI U gross RCC trên 43 ảnh train, đánh giá
trên 12 ảnh test (labels/split.json). Tập NHỎ -> chỉ train prompt_encoder +
mask_decoder, ĐÓNG BĂNG image encoder (đặc trưng tính 1 lần dưới no_grad).

Mỗi mask u (labels/masks/<stem>.png) tách connected-component -> mỗi cụm = 1 mẫu;
prompt = bbox cụm (jitter ngẫu nhiên khi train). Loss = Dice + BCE ở 256x256.
Eval = Dice mask đầy đủ (predictor.predict với bbox sát, không jitter).

  python finetune_sam2.py --epochs 60 --lr 1e-4
Lưu best theo test Dice -> checkpoints/sam2.1_rcc_ft.pt ; log results/finetune_log.csv
"""
import argparse
import csv
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

IMG_DIR = os.path.join(ROOT, "data/20241212")
LABELS = os.path.join(ROOT, "labels")
CONFIG = os.environ.get("SAM2_CONFIG", "configs/sam2.1_hiera_t512")
CKPT = os.environ.get("SAM2_CKPT", "checkpoints/sam2.1_hiera_tiny.pt")
RES = int(os.environ.get("SAM2_RES", "1024"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AC = torch.autocast("cuda", dtype=torch.bfloat16)
SEED = 0


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def read_rgb(stem):
    return cv2.cvtColor(cv2.imread(os.path.join(IMG_DIR, stem + ".jpg")), cv2.COLOR_BGR2RGB)


def read_mask(stem):
    m = cv2.imread(os.path.join(LABELS, "masks", stem + ".png"), cv2.IMREAD_GRAYSCALE)
    return (m > 127).astype(np.uint8)


def components(mask, min_frac=0.001):
    """Tách cụm; bỏ cụm < min_frac diện tích ảnh."""
    n, lab = cv2.connectedComponents(mask, connectivity=8)
    out = []
    thr = min_frac * mask.size
    for i in range(1, n):
        c = (lab == i)
        if c.sum() >= thr:
            out.append(c)
    if not out and mask.sum() > 0:
        out = [mask.astype(bool)]
    return out


def bbox_of(m):
    ys, xs = np.where(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def jitter_box(box, W, H, frac=0.1):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    dx = random.uniform(-frac, frac) * bw
    dy = random.uniform(-frac, frac) * bh
    ex = random.uniform(0, frac) * bw
    ey = random.uniform(0, frac) * bh
    return [max(0, x0 + dx - ex), max(0, y0 + dy - ey),
            min(W - 1, x1 + dx + ex), min(H - 1, y1 + dy + ey)]


def decode(model, feats, orig_hw, box, transforms):
    """Mirror _predict cho 1 box -> low_res logits (1,1,256,256)."""
    b = torch.as_tensor([box], dtype=torch.float, device=DEVICE)
    bt = transforms.transform_boxes(b, normalize=True, orig_hw=orig_hw)
    box_coords = bt.reshape(-1, 2, 2)
    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=DEVICE)
    sparse, dense = model.sam_prompt_encoder(
        points=(box_coords, box_labels), boxes=None, masks=None)
    low_res, iou, _, _ = model.sam_mask_decoder(
        image_embeddings=feats["image_embed"],
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        high_res_features=list(feats["high_res_feats"]),
    )
    return low_res  # (1,1,256,256) logits


def dice_bce(logits, target):
    """logits,target: (1,1,h,w). target {0,1} float."""
    p = torch.sigmoid(logits.float())
    t = target.float()
    inter = (p * t).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + t.sum() + 1)
    bce = F.binary_cross_entropy_with_logits(logits.float(), t)
    return dice + bce


def decode_batch(model, image_embed, high_res_feats, orig_hw, boxes, transforms):
    """Batched decode: B ảnh (cùng orig_hw) mỗi ảnh 1 box -> (B,1,256,256) logits."""
    b = torch.as_tensor(boxes, dtype=torch.float, device=DEVICE)            # (B,4)
    bt = transforms.transform_boxes(b, normalize=True, orig_hw=orig_hw)     # (B,4)
    box_coords = bt.reshape(-1, 2, 2)                                       # (B,2,2)
    B = box_coords.shape[0]
    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=DEVICE).repeat(B, 1)
    sparse, dense = model.sam_prompt_encoder(
        points=(box_coords, box_labels), boxes=None, masks=None)
    low_res, iou, _, _ = model.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_feats,
    )
    return low_res  # (B,1,256,256)


def dice_bce_batch(logits, target):
    """logits,target: (B,1,h,w). Dice+BCE trung bình theo batch."""
    p = torch.sigmoid(logits.float())
    t = target.float()
    dims = (1, 2, 3)
    inter = (p * t).sum(dims)
    dice = 1 - (2 * inter + 1) / (p.sum(dims) + t.sum(dims) + 1)
    bce = F.binary_cross_entropy_with_logits(
        logits.float(), t, reduction="none").mean(dims)
    return (dice + bce).mean()


class _PrecomputeDS(torch.utils.data.Dataset):
    """Phần CPU của precompute (đọc JPG + mask + tách cụm) để DataLoader chạy song song."""
    def __init__(self, stems):
        self.stems = stems

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        s = self.stems[i]
        gt = read_mask(s)
        H, W = gt.shape
        return s, read_rgb(s), components(gt), (W, H)


@torch.no_grad()
def eval_dice(predictor, stems):
    predictor.model.eval()
    dices = []
    for s in stems:
        rgb = read_rgb(s)
        gt = read_mask(s)
        with AC:
            predictor.set_image(rgb)
        comp = components(gt)
        for c in comp:
            box = np.array(bbox_of(c), dtype=np.float32)
            with AC:
                masks, scores, _ = predictor.predict(
                    box=box, multimask_output=False)
            pred = masks[0].astype(bool)
            inter = (pred & c).sum()
            d = 2 * inter / (pred.sum() + c.sum() + 1e-9)
            dices.append(float(d))
    return float(np.mean(dices)) if dices else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--precompute_batch", type=int, default=16,
                    help="số ảnh/lần qua image-encoder khi precompute (batch GPU -> nhanh)")
    ap.add_argument("--split", default=os.path.join(LABELS, "split.json"),
                    help="file split.json {train,val} (đổi để train HELD-OUT)")
    ap.add_argument("--ckpt_out", default=os.path.join(ROOT, "checkpoints/sam2.1_rcc_ft.pt"),
                    help="đường dẫn lưu checkpoint FT")
    args = ap.parse_args()
    set_seed(SEED)

    split = json.load(open(args.split))
    train_stems, test_stems = split["train"], split["val"]
    print(f"Train {len(train_stems)} ảnh | Test {len(test_stems)} ảnh")

    print(f"Nạp SAM2.1@{RES} trên {DEVICE} ...")
    model = build_sam2(CONFIG, CKPT, device=DEVICE,
                       hydra_overrides_extra=[f"++model.image_size={RES}"])
    predictor = SAM2ImagePredictor(model)
    transforms = predictor._transforms

    # đóng băng image encoder; train prompt_encoder + mask_decoder
    for p in model.parameters():
        p.requires_grad_(False)
    train_mods = [model.sam_prompt_encoder, model.sam_mask_decoder]
    params = []
    for m in train_mods:
        for p in m.parameters():
            p.requires_grad_(True)
            params.append(p)
    print(f"Số tham số train: {sum(p.numel() for p in params)/1e6:.2f}M")

    # precompute đặc trưng encoder (no_grad) + cụm mask cho train -> cache CPU
    # đọc JPG/mask song song bằng DataLoader (workers); encoder vẫn chạy GPU tuần tự
    print(f"Trích đặc trưng encoder cho train (BATCH GPU={args.precompute_batch}, "
          f"workers={args.workers})...")
    loader = torch.utils.data.DataLoader(
        _PrecomputeDS(train_stems), batch_size=args.precompute_batch,
        num_workers=args.workers, collate_fn=lambda b: b,
        persistent_workers=False)
    def precompute_chunk(items):
        """encoder cho 1 nhóm ảnh; OOM -> tự CHIA ĐÔI (bền với GPU dùng chung)."""
        try:
            with AC:
                predictor.set_image_batch([it[1] for it in items])
            ie = predictor._features["image_embed"]
            hrf = predictor._features["high_res_feats"]
            out = []
            for i, (s, rgb, comps, (W, H)) in enumerate(items):
                feats = {"image_embed": ie[i:i + 1].detach().to("cpu"),
                         "high_res_feats": [f[i:i + 1].detach().to("cpu") for f in hrf]}
                orig_hw = tuple(predictor._orig_hw[i])
                for c in comps:
                    out.append({"feats": feats, "orig_hw": orig_hw, "comp": c, "WH": (W, H)})
            return out
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(items) == 1:
                raise
            mid = len(items) // 2
            return precompute_chunk(items[:mid]) + precompute_chunk(items[mid:])

    cache = []
    for batch in loader:                       # batch = list các (s, rgb, comps, (W,H))
        cache.extend(precompute_chunk(batch))
    print(f"Mẫu train (cụm u): {len(cache)}")

    # gom chỉ số mẫu theo orig_hw để batch (ảnh khác kích thước nằm batch riêng)
    from collections import defaultdict
    groups = defaultdict(list)
    for i, it in enumerate(cache):
        groups[it["orig_hw"]].append(i)
    print(f"Nhóm kích thước: {[(k, len(v)) for k, v in groups.items()]} | batch={args.batch}")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    logp = os.path.join(ROOT, "results/finetune_log.csv")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    logf = open(logp, "w", newline=""); logw = csv.writer(logf)
    logw.writerow(["epoch", "train_loss", "test_dice"]); logf.flush()

    base_dice = eval_dice(predictor, test_stems)
    print(f"[epoch 0] zero-shot test Dice = {base_dice:.4f}")
    logw.writerow([0, "", round(base_dice, 4)]); logf.flush()

    best = base_dice          # để báo cáo so với zero-shot
    best_ft = -1.0            # test-Dice tốt nhất CỦA FT (LƯU bất kể vượt zero-shot hay không)
    best_path = args.ckpt_out
    for ep in range(1, args.epochs + 1):
        model.train()
        # dựng danh sách batch: trong mỗi nhóm orig_hw, cắt theo args.batch
        order = []
        for k, idxs in groups.items():
            idxs = idxs[:]; random.shuffle(idxs)
            for i in range(0, len(idxs), args.batch):
                order.append(idxs[i:i + args.batch])
        random.shuffle(order)
        tot = 0.0; nb = 0
        for bidx in order:
            items = [cache[i] for i in bidx]
            image_embed = torch.stack(
                [it["feats"]["image_embed"][0] for it in items]).to(DEVICE)
            n_lvl = len(items[0]["feats"]["high_res_feats"])
            high_res_feats = [
                torch.stack([it["feats"]["high_res_feats"][lvl][0] for it in items]).to(DEVICE)
                for lvl in range(n_lvl)]
            orig_hw = items[0]["orig_hw"]
            boxes = [jitter_box(bbox_of(it["comp"]), *it["WH"]) for it in items]
            tg = []
            for it in items:
                t = cv2.resize(it["comp"].astype(np.float32), (256, 256),
                               interpolation=cv2.INTER_AREA)
                tg.append((t > 0.5).astype(np.float32))
            target = torch.as_tensor(np.stack(tg), device=DEVICE)[:, None]
            with AC:
                logits = decode_batch(model, image_embed, high_res_feats,
                                      orig_hw, boxes, transforms)
            loss = dice_bce_batch(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        tl = tot / max(nb, 1)
        msg = f"[epoch {ep}] train_loss={tl:.4f}"
        td = ""
        if ep % args.eval_every == 0 or ep == args.epochs:
            td = eval_dice(predictor, test_stems)
            msg += f" | test Dice={td:.4f}"
            best = max(best, td)
            if td > best_ft:           # LUÔN lưu epoch FT tốt nhất (ít overfit nhất)
                best_ft = td
                torch.save({"model": model.state_dict(), "test_dice": td, "epoch": ep,
                            "zero_shot_dice": base_dice,
                            "beats_zero_shot": bool(td > base_dice)}, best_path)
                msg += "  *best-FT -> lưu"
        print(msg)
        logw.writerow([ep, round(tl, 4), round(td, 4) if td != "" else ""]); logf.flush()

    logf.close()
    json.dump({"n_train": len(train_stems), "n_train_comp": len(cache),
               "n_test": len(test_stems), "zero_shot_dice": round(base_dice, 4),
               "best_ft_dice": round(best_ft, 4), "epochs": args.epochs,
               "beats_zero_shot": bool(best_ft > base_dice),
               "ckpt": best_path}, open(os.path.join(ROOT, "results/finetune_last.json"), "w"), indent=1)
    print(f"\nXong. zero-shot Dice={base_dice:.4f} | BEST fine-tuned Dice={best_ft:.4f} "
          f"({'VƯỢT' if best_ft > base_dice else 'KHÔNG vượt'} zero-shot)")
    print(f"Checkpoint LƯU: {best_path} (epoch test-Dice cao nhất) | log: {logp}")


if __name__ == "__main__":
    main()
