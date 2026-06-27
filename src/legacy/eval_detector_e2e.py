"""
Eval END-TO-END detector DENSE trên eval200 (cùng tập, cùng cách như e2e cũ):
  ảnh -> SAM encoder -> detector tự đề xuất N box (objectness>thr) -> SAM predict
  (multimask, best score) -> GỘP mask -> Dice/HD95 vs mask tay.
So sánh với pipeline cellbox cũ (results/e2e_ft_vs_zs.csv, cột dice_zs).
Chọn objectness-thr trên val. Xuất results/detector_e2e.csv + results/detector_montage.png
"""
import os, sys, json, csv, numpy as np, cv2, torch
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from detector import DenseDetector, decode_detections, cxcywh_to_xyxy, propose_boxes
from specimen_clean import clean_specimen
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CFG = "configs/sam2.1_hiera_t512"; CKPT = "checkpoints/sam2.1_hiera_tiny.pt"
RES = 1024; AC = torch.autocast("cuda", dtype=torch.bfloat16)
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


def detect(det, predictor, rgb, thr, spec=None):
    """-> (boxes_px list[xyxy], union_mask). Gate thước/nhãn + fallback recall đa-u."""
    H, W = rgb.shape[:2]
    with torch.inference_mode(), AC:
        predictor.set_image(rgb)
        feat = predictor._features["image_embed"].float()
        obj, boxes = det(feat)
    px = propose_boxes(obj[0].float(), boxes[0].float(), H, W, spec=spec, thr=thr)
    union = np.zeros((H, W), bool)
    with torch.inference_mode(), AC:
        for b in px:
            mk, sc, _ = predictor.predict(box=b, multimask_output=True)
            union |= mk[int(np.argmax(sc))].astype(bool)
    return px, union


def run(stems, det, predictor, thr):
    out = {}
    for s in stems:
        bgr = cv2.imread(f"data/20241212/{s}.jpg"); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        spec, _, _ = clean_specimen(bgr)
        gt = cv2.imread(f"labels/masks/{s}.png", 0) > 127
        px, mask = detect(det, predictor, rgb, thr, spec=spec)
        out[s] = (dice(mask, gt), hd95(mask, gt), len(px), mask, px)
    return out


def nobj(s):
    p = f"labels/prompts/{s}.json"
    return json.load(open(p)).get("n_objects", 1) if os.path.isfile(p) else 1


def main():
    DET_CKPT = os.environ.get("DET_CKPT", "checkpoints/detector.pt")
    TAG = os.environ.get("DET_TAG", "")
    CSV_OUT = f"results/detector_e2e{TAG}.csv"; MON_OUT = f"results/detector_montage{TAG}.png"
    print(f"[ckpt={DET_CKPT} tag='{TAG}'] -> {CSV_OUT}", flush=True)
    sp = json.load(open("labels/split_detector.json"))
    val, eval200 = sp["val"], sp["eval200"]
    ck = torch.load(DET_CKPT, weights_only=False)
    det = DenseDetector(grid=ck.get("grid", 64)).to("cuda"); det.load_state_dict(ck["det"]); det.eval()
    print(f"detector epoch={ck['epoch']} val_dice(train-time)={ck['val_dice']:.4f}@thr{ck.get('obj_thr','?')}", flush=True)
    model = build_sam2(CFG, CKPT, device="cuda", hydra_overrides_extra=[f"++model.image_size={RES}"])
    predictor = SAM2ImagePredictor(model)

    # chọn objectness-thr trên val
    print("Quét objectness-thr trên val...", flush=True)
    best_thr, best_d = 0.5, -1
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        r = run(val, det, predictor, thr)
        md = np.median([v[0] for v in r.values()])
        print(f"  thr={thr}: val Dice median={md:.4f} | box/ảnh tv={np.median([v[2] for v in r.values()]):.0f}", flush=True)
        if md > best_d: best_d, best_thr = md, thr
    print(f"-> chọn thr={best_thr}", flush=True)

    # eval200
    print(f"Eval eval200 (N={len(eval200)}) thr={best_thr}...", flush=True)
    R = run(eval200, det, predictor, best_thr)
    D = np.array([R[s][0] for s in eval200]); HDv = np.array([R[s][1] for s in eval200])
    NB = np.array([R[s][2] for s in eval200]); NO = np.array([nobj(s) for s in eval200])

    # số cũ (cellbox) trên cùng stem
    old = {r["stem"]: float(r["dice_zs"]) for r in csv.DictReader(open("results/e2e_ft_vs_zs.csv"))}
    OZ = np.array([old.get(s, np.nan) for s in eval200])

    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stem", "dice_detector", "hd95", "n_box", "n_obj", "dice_old_cellbox_zs"])
        for s in eval200:
            w.writerow([s, round(R[s][0], 4), round(R[s][1], 2) if not np.isnan(R[s][1]) else "",
                        R[s][2], nobj(s), round(old.get(s, np.nan), 4) if s in old else ""])
    print(f"\n===== DETECTOR DENSE end-to-end, N={len(eval200)}, thr={best_thr} =====")
    print(f"box/ảnh trung vị={np.median(NB):.0f} | tìm >=1 box: {(NB>0).sum()}/{len(NB)}")
    print(f"Detector  Dice median={np.median(D):.4f} mean={np.mean(D):.4f}")
    print(f"Cellbox(cũ,ZS) Dice median={np.nanmedian(OZ):.4f} mean={np.nanmean(OZ):.4f}")
    print(f"  1 u  (n={int((NO<=1).sum())}): DENSE={np.median(D[NO<=1]):.3f} | cũ={np.nanmedian(OZ[NO<=1]):.3f}")
    print(f"  >1 u (n={int((NO>1).sum())}): DENSE={np.median(D[NO>1]):.3f} | cũ={np.nanmedian(OZ[NO>1]):.3f}")
    paired = ~np.isnan(OZ)
    print(f"DENSE thắng cellbox: {(D[paired]>OZ[paired]).sum()}/{paired.sum()}  "
          f"delta median={np.median(D[paired]-OZ[paired]):+.4f}")
    print(f"-> {CSV_OUT}", flush=True)

    # montage: 18 ca Dice thấp nhất + 6 cao nhất
    order = sorted(range(len(eval200)), key=lambda i: D[i]); sub = order[:18] + order[-6:]
    PW = 340; tiles = []
    for i in sub:
        s = eval200[i]; gt = cv2.imread(f"labels/masks/{s}.png", 0) > 127
        rgb = cv2.cvtColor(cv2.imread(f"data/20241212/{s}.jpg"), cv2.COLOR_BGR2RGB)
        ys, xs = np.where(gt); m = 160
        sl = (slice(max(0, ys.min()-m), min(rgb.shape[0], ys.max()+m)),
              slice(max(0, xs.min()-m), min(rgb.shape[1], xs.max()+m)))
        def pan(img, lab, col):
            Hh = int(PW*img.shape[0]/img.shape[1]); im = cv2.resize(img, (PW, Hh))
            bar = np.full((26, PW, 3), 25, np.uint8); cv2.putText(bar, lab, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            return np.vstack([bar, im])
        o = rgb.copy(); mask = R[s][3]
        if mask.any(): o[mask] = (0.45*o[mask] + 0.55*np.array([255, 40, 40])).astype(np.uint8)
        for b in R[s][4]: cv2.rectangle(o, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 120, 255), 3)
        gc_, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, gc_, -1, (255, 255, 0), 4)
        p_o = pan(rgb[sl], f"GOC ({NO[i]}u)", (255, 255, 255))
        p_p = pan(o[sl], f"DENSE d={D[i]:.3f} box={R[s][2]}", (120, 120, 255))
        h = max(p_o.shape[0], p_p.shape[0])
        pad = lambda p: np.vstack([p, np.full((h-p.shape[0], PW, 3), 25, np.uint8)])
        sep = np.full((h, 8, 3), 60, np.uint8)
        tiles.append(np.hstack([pad(p_o), sep, pad(p_p)]))
    COLS = 3; grows = []
    for i in range(0, len(tiles), COLS):
        rw = tiles[i:i+COLS]; h = max(c.shape[0] for c in rw)
        rw = [np.vstack([c, np.full((h-c.shape[0], c.shape[1], 3), 15, np.uint8)]) for c in rw]
        while len(rw) < COLS: rw.append(np.full((h, rw[0].shape[1], 3), 15, np.uint8))
        g = np.full((h, 16, 3), 15, np.uint8); o = rw[0]
        for c in rw[1:]: o = np.hstack([o, g, c])
        grows.append(o)
    W = max(r.shape[1] for r in grows)
    grows = [np.hstack([r, np.full((r.shape[0], W-r.shape[1], 3), 15, np.uint8)]) for r in grows]
    grid = np.vstack(grows)
    leg = np.full((46, W, 3), 0, np.uint8)
    cv2.putText(leg, "DETECTOR DENSE auto multi-box -> SAM. GOC | DENSE(do): khung xanh=box tu de xuat, vien VANG=GT. 18 ca kem + 6 tot.",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    grid = np.vstack([leg, grid])
    cv2.imwrite(MON_OUT, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 4])
    print(f"-> {MON_OUT}", grid.shape, flush=True)


if __name__ == "__main__":
    main()
