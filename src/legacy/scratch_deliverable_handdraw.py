"""
DELIVERABLE SẠCH n=50: box GT = bbox TỪNG MẢNH (connected component) của MASK VẼ TAY
(ground-truth độc lập) -> SAM2.1 zero-shot -> union -> Dice/HD95 vs chính mask vẽ tay.
Đo TRẦN box->mask khi localize hoàn hảo, KHÔNG phụ thuộc detector. Thay con số n=8 cũ
(labels/prompts box, theo mảnh, thiên vị) bằng n=50 trên đúng ground-truth.
Chạy: /home/hvusynh2/conda_envs/medsam2_anno/bin/python -u scratch_deliverable_handdraw.py
"""
import os, sys, json, csv, numpy as np, cv2, torch
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CFG = "configs/sam2.1_hiera_t512"; CKPT = "checkpoints/sam2.1_hiera_tiny.pt"
RES = 1024; AC = torch.autocast("cuda", dtype=torch.bfloat16)
HMASK = "labels_handdraw/masks"; MIN_FRAC = 0.002

def dice(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s
def hd95(pred, gt):
    def bd(m): mu = m.astype(np.uint8); return (mu - cv2.erode(mu, np.ones((3, 3), np.uint8))) > 0
    pb, gb = bd(pred), bd(gt)
    if pb.sum() == 0 or gb.sum() == 0: return np.nan
    dg = cv2.distanceTransform((~gb).astype(np.uint8), cv2.DIST_L2, 3)
    dp = cv2.distanceTransform((~pb).astype(np.uint8), cv2.DIST_L2, 3)
    return float(np.percentile(np.concatenate([dg[pb], dp[gb]]), 95))

def gt_boxes(mask):
    """bbox mỗi connected component >= MIN_FRAC diện tích -> list [x0,y0,x1,y1]."""
    m = mask.astype(np.uint8); n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    thr = MIN_FRAC * m.size; boxes = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < thr: continue
        x, y, w, h = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        boxes.append([float(x), float(y), float(x + w - 1), float(y + h - 1)])
    return boxes

def sam_union(predictor, rgb, boxes):
    H, W = rgb.shape[:2]; union = np.zeros((H, W), bool)
    with torch.inference_mode(), AC:
        predictor.set_image(rgb)
        for b in boxes:
            mk, sc, _ = predictor.predict(box=np.array(b, np.float32), multimask_output=True)
            union |= mk[int(np.argmax(sc))].astype(bool)
    return union

def main():
    stems = json.load(open("labels_handdraw/select.json"))["stems"]
    have = [s for s in stems if os.path.isfile(f"{HMASK}/{s}.png")]
    print(f"mask vẽ tay: {len(have)}/{len(stems)}", flush=True)
    model = build_sam2(CFG, CKPT, device="cuda", hydra_overrides_extra=[f"++model.image_size={RES}"])
    predictor = SAM2ImagePredictor(model)
    rows = []
    for s in have:
        gt = cv2.imread(f"{HMASK}/{s}.png", 0) > 127
        bgr = cv2.imread(f"data/20241212/{s}.jpg"); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bx = gt_boxes(gt); n_obj = max(1, len(bx))
        mk = sam_union(predictor, rgb, bx) if bx else np.zeros_like(gt)
        d = dice(mk, gt); h = hd95(mk, gt)
        rows.append(dict(stem=s, n_obj=n_obj, n_box=len(bx), dice=d, hd95=h))
        print(f"  {s[:22]} n_u={n_obj} box={len(bx)} dice={d:.3f}", flush=True)
    with open("results/handdraw_deliverable.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stem", "n_obj", "n_box", "dice", "hd95"])
        for r in rows: w.writerow([r["stem"], r["n_obj"], r["n_box"], round(r["dice"], 4),
                                   round(r["hd95"], 2) if r["hd95"] == r["hd95"] else ""])
    def med(f=lambda r: True):
        v = np.array([r["dice"] for r in rows if f(r)]); return np.median(v), np.mean(v), len(v)
    one = lambda r: r["n_obj"] <= 1; mul = lambda r: r["n_obj"] > 1
    a, o, m = med(), med(one), med(mul)
    hh = np.array([r["hd95"] for r in rows]); hh = hh[~np.isnan(hh)]
    print(f"\n===== DELIVERABLE SẠCH SAM(GT-box từ mask vẽ tay), N={len(rows)} =====")
    print(f"Dice median={a[0]:.4f} mean={a[1]:.4f} | 1u(n={o[2]})={o[0]:.3f} >1u(n={m[2]})={m[0]:.3f} | HD95 median={np.median(hh):.1f}px")
    print("-> results/handdraw_deliverable.csv", flush=True)

if __name__ == "__main__":
    main()
