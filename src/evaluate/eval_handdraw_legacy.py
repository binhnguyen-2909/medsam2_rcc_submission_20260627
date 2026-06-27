"""
EVAL KHÁCH QUAN trên TEST SET VẼ TAY (labels_handdraw/masks) — phá vòng GT-do-SAM.
Chỉ đánh giá các stem ĐÃ có mask vẽ tay. Với mỗi stem, đo Dice/HD95 vs mask VẼ TAY của:
  (1) SAM-made cũ (labels/masks)      -> ĐO ĐỘ THIÊN VỊ của nhãn cũ (không cần model)
  (2) Deliverable: SAM(box=GT-bbox)   -> chất lượng box->mask thật (cận trên)
  (3) Full-auto: DenseDetector+maskloss -> SAM (union)  -> pipeline tự động thật
Tách 1u/>1u theo #component của MASK VẼ TAY. Xuất results/handdraw_eval.csv + montage.
Chạy: /home/hvusynh2/conda_envs/medsam2_anno/bin/python -u eval_handdraw.py
"""
import os, sys, json, csv, numpy as np, cv2, torch
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from detector import DenseDetector, decode_detections, cxcywh_to_xyxy, propose_boxes
from specimen_clean import clean_specimen
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CFG = "configs/sam2.1_hiera_t512"; CKPT = "checkpoints/sam2.1_hiera_tiny.pt"
DET_CKPT = os.environ.get("DET_CKPT", "checkpoints/detector_maskloss.pt")
DET_THR = float(os.environ.get("DET_THR", "0.5"))
RES = 1024; AC = torch.autocast("cuda", dtype=torch.bfloat16)
HMASK = "labels_handdraw/masks"; SAM_MASK = "labels/masks"; PROMPT = "labels/prompts"
MIN_FRAC = 0.002

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
def ncomp(m):
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return max(1, int((st[1:, cv2.CC_STAT_AREA] >= MIN_FRAC * m.size).sum()))

def sam_from_boxes(predictor, rgb, boxes_px):
    H, W = rgb.shape[:2]; union = np.zeros((H, W), bool)
    with torch.inference_mode(), AC:
        predictor.set_image(rgb)
        for b in boxes_px:
            mk, sc, _ = predictor.predict(box=b, multimask_output=True)
            union |= mk[int(np.argmax(sc))].astype(bool)
    return union

def auto_boxes(det, predictor, rgb, thr, spec=None):
    """Đề xuất box tự động: gate thước/nhãn (spec) + fallback recall đa-u (propose_boxes)."""
    H, W = rgb.shape[:2]
    with torch.inference_mode(), AC:
        predictor.set_image(rgb)
        feat = predictor._features["image_embed"].float()
        obj, boxes = det(feat)
    return propose_boxes(obj[0].float(), boxes[0].float(), H, W, spec=spec, thr=thr)

def main():
    stems = json.load(open("labels_handdraw/select.json"))["stems"]
    have = [s for s in stems if os.path.isfile(f"{HMASK}/{s}.png")]
    print(f"Stem có mask vẽ tay: {len(have)}/{len(stems)}", flush=True)
    if not have:
        print("CHƯA có mask vẽ tay nào — vẽ trong app trước (cổng 18864)."); return

    model = build_sam2(CFG, CKPT, device="cuda", hydra_overrides_extra=[f"++model.image_size={RES}"])
    predictor = SAM2ImagePredictor(model)
    ck = torch.load(DET_CKPT, weights_only=False)
    det = DenseDetector(grid=ck.get("grid", 64)).to("cuda"); det.load_state_dict(ck["det"]); det.eval()
    print(f"detector={DET_CKPT} ep{ck['epoch']} thr={DET_THR}", flush=True)

    rows = []
    for s in have:
        gt = cv2.imread(f"{HMASK}/{s}.png", 0) > 127
        bgr = cv2.imread(f"data/20241212/{s}.jpg"); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        spec, _, _ = clean_specimen(bgr)
        no = ncomp(gt)
        # (1) SAM-made cũ vs vẽ tay
        d_old = h_old = np.nan
        if os.path.isfile(f"{SAM_MASK}/{s}.png"):
            old = cv2.imread(f"{SAM_MASK}/{s}.png", 0) > 127
            d_old = dice(old, gt); h_old = hd95(old, gt)
        # (2) deliverable SAM(GT-box)
        d_box = h_box = np.nan
        if os.path.isfile(f"{PROMPT}/{s}.json"):
            pj = json.load(open(f"{PROMPT}/{s}.json")); bx = pj.get("box")
            if bx:
                mk = sam_from_boxes(predictor, rgb, [np.array(bx, np.float32)])
                d_box = dice(mk, gt); h_box = hd95(mk, gt)
        # (3) full-auto detector+maskloss (gate thước/nhãn + fallback recall đa-u)
        ab = auto_boxes(det, predictor, rgb, DET_THR, spec=spec)
        mk_auto = sam_from_boxes(predictor, rgb, ab) if len(ab) else np.zeros_like(gt)
        d_auto = dice(mk_auto, gt); h_auto = hd95(mk_auto, gt)
        rows.append(dict(stem=s, n_obj=no, n_box_auto=len(ab),
                         dice_samGT_vs_hand=d_old, hd95_samGT=h_old,
                         dice_deliverable_box=d_box, hd95_box=h_box,
                         dice_auto_maskloss=d_auto, hd95_auto=h_auto,
                         _auto_mask=mk_auto, _gt=gt, _rgb=rgb, _ab=ab))
        print(f"  {s[:22]} n_u={no} | SAMcũ={d_old:.3f} box={d_box:.3f} auto={d_auto:.3f}", flush=True)

    # ---- tổng hợp ----
    def med(key, mask=None):
        v = np.array([r[key] for r in rows if (mask is None or mask(r))], float)
        v = v[~np.isnan(v)]; return (np.median(v), np.mean(v), len(v))
    with open("results/handdraw_eval.csv", "w", newline="") as f:
        cols = ["stem", "n_obj", "n_box_auto", "dice_samGT_vs_hand", "hd95_samGT",
                "dice_deliverable_box", "hd95_box", "dice_auto_maskloss", "hd95_auto"]
        w = csv.writer(f); w.writerow(cols)
        for r in rows: w.writerow([round(r[c], 4) if isinstance(r[c], float) else r[c] for c in cols])
    one = lambda r: r["n_obj"] <= 1; mul = lambda r: r["n_obj"] > 1
    n1 = sum(1 for r in rows if one(r)); nm = sum(1 for r in rows if mul(r))
    print(f"\n===== EVAL VẼ TAY (N={len(rows)}: {n1} đơn-u + {nm} đa-u) =====")
    for key, lab in [("dice_samGT_vs_hand", "SAM-cũ vs vẽtay (độ THIÊN VỊ nhãn cũ)"),
                     ("dice_deliverable_box", "Deliverable SAM(GT-box)"),
                     ("dice_auto_maskloss", "Full-auto detector+maskloss")]:
        a = med(key); o = med(key, one); m = med(key, mul)
        print(f"{lab:42s} Dice median={a[0]:.4f} mean={a[1]:.4f} (n={a[2]}) | "
              f"1u={o[0]:.3f} >1u={m[0]:.3f}")
    print("-> results/handdraw_eval.csv", flush=True)

    # montage: tất cả (≤24) hoặc 24 ca auto thấp nhất
    rows_s = sorted(rows, key=lambda r: r["dice_auto_maskloss"])[:24]
    PW = 320; tiles = []
    for r in rows_s:
        gt = r["_gt"]; rgb = r["_rgb"]; ys, xs = np.where(gt)
        if len(ys) == 0: continue
        mm = 160; sl = (slice(max(0, ys.min()-mm), min(rgb.shape[0], ys.max()+mm)),
                        slice(max(0, xs.min()-mm), min(rgb.shape[1], xs.max()+mm)))
        def pan(img, lab, col):
            Hh = int(PW*img.shape[0]/img.shape[1]); im = cv2.resize(img, (PW, Hh))
            bar = np.full((24, PW, 3), 25, np.uint8); cv2.putText(bar, lab, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            return np.vstack([bar, im])
        o = rgb.copy(); mb = r["_auto_mask"]
        if mb.any(): o[mb] = (0.45*o[mb] + 0.55*np.array([255, 40, 40])).astype(np.uint8)
        for b in r["_ab"]: cv2.rectangle(o, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 120, 255), 3)
        gc, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, gc, -1, (255, 255, 0), 4)
        p1 = pan(rgb[sl], f"GOC ({r['n_obj']}u)", (255, 255, 255))
        p2 = pan(o[sl], f"auto d={r['dice_auto_maskloss']:.2f} box={r['n_box_auto']}", (120, 120, 255))
        h = max(p1.shape[0], p2.shape[0]); pad = lambda p: np.vstack([p, np.full((h-p.shape[0], PW, 3), 25, np.uint8)])
        tiles.append(np.hstack([pad(p1), np.full((h, 6, 3), 60, np.uint8), pad(p2)]))
    if tiles:
        COLS = 3; rws = []
        for i in range(0, len(tiles), COLS):
            rw = tiles[i:i+COLS]; hh = max(c.shape[0] for c in rw)
            rw = [np.vstack([c, np.full((hh-c.shape[0], c.shape[1], 3), 15, np.uint8)]) for c in rw]
            while len(rw) < COLS: rw.append(np.full((hh, rw[0].shape[1], 3), 15, np.uint8))
            g = np.full((hh, 12, 3), 15, np.uint8); o = rw[0]
            for c in rw[1:]: o = np.hstack([o, g, c])
            rws.append(o)
        Wd = max(r.shape[1] for r in rws)
        rws = [np.hstack([r, np.full((r.shape[0], Wd-r.shape[1], 3), 15, np.uint8)]) for r in rws]
        cv2.imwrite("results/handdraw_montage.png", cv2.cvtColor(np.vstack(rws), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 4])
        print("-> results/handdraw_montage.png", flush=True)

if __name__ == "__main__":
    main()
