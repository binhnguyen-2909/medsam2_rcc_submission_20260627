"""
PHÂN RÃ LỖI ĐA-U trên test VẼ TAY: với mỗi MẢNH u (connected component của mask vẽ tay),
xác định lỗi nằm ở đâu:
  (R) RECALL  : mảnh u có được detector phủ box không? (>=COVER pixel mảnh nằm trong union box)
  (S) SEGMENT : nếu cho SAM box=bbox-TAY của ĐÚNG mảnh đó, Dice vs mảnh = trần segment (localize hoàn hảo)
  (F) FALSE+  : box detector có TÂM trong bệnh phẩm nhưng KHÔNG trùng mảnh u nào = dương tính giả
So với full-auto thật (propose_boxes: gate+fallback). Trả lời: nút thắt đa-u là RECALL hay SEGMENT?
Chạy nền: .../python -u scratch_handdraw_diag.py > results/handdraw_diag.log 2>&1 &
"""
import os, sys, json, csv, numpy as np, cv2, torch
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from detector import DenseDetector, propose_boxes
from specimen_clean import clean_specimen
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CFG = "configs/sam2.1_hiera_t512"; CKPT = "checkpoints/sam2.1_hiera_tiny.pt"
DET_CKPT = os.environ.get("DET_CKPT", "checkpoints/detector_maskloss.pt")
RES = 1024; AC = torch.autocast("cuda", dtype=torch.bfloat16)
HMASK = "labels_handdraw/masks"; MIN_FRAC = 0.002; COVER = 0.30

def dice(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s

def comps(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= MIN_FRAC * m.size:
            x, y, w, h, _ = st[i]
            out.append((lab == i, (x, y, x + w, y + h)))
    return out

def sam_one(predictor, box):
    with torch.inference_mode(), AC:
        mk, sc, _ = predictor.predict(box=np.array(box, np.float32), multimask_output=True)
    return mk[int(np.argmax(sc))].astype(bool)

def sam_union(predictor, boxes, shape):
    u = np.zeros(shape, bool)
    for b in boxes: u |= sam_one(predictor, b)
    return u

def main():
    stems = json.load(open("labels_handdraw/select.json"))["stems"]
    have = [s for s in stems if os.path.isfile(f"{HMASK}/{s}.png")]
    model = build_sam2(CFG, CKPT, device="cuda", hydra_overrides_extra=[f"++model.image_size={RES}"])
    predictor = SAM2ImagePredictor(model)
    ck = torch.load(DET_CKPT, weights_only=False)
    det = DenseDetector(grid=ck.get("grid", 64)).to("cuda"); det.load_state_dict(ck["det"]); det.eval()
    print(f"detector={DET_CKPT} ep{ck['epoch']} | COVER={COVER}", flush=True)

    rows = []
    for s in have:
        gt = cv2.imread(f"{HMASK}/{s}.png", 0) > 127
        cs = comps(gt)
        if len(cs) <= 1:   # chỉ quan tâm đa-u
            continue
        bgr = cv2.imread(f"data/20241212/{s}.jpg"); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = gt.shape; spec, _, _ = clean_specimen(bgr)
        with torch.inference_mode(), AC:
            predictor.set_image(rgb)
            feat = predictor._features["image_embed"].float()
            obj, boxes = det(feat)
        det_px = propose_boxes(obj[0].float(), boxes[0].float(), H, W, spec=spec, thr=0.5)
        # union các box detector (vùng phủ)
        boxcov = np.zeros((H, W), bool)
        for b in det_px:
            x0, y0, x1, y1 = [int(v) for v in b]
            boxcov[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
        # (R) recall theo mảnh + (S) trần segment khi box=bbox-tay đúng mảnh
        detected = 0; seg_dices = []; auto_full = sam_union(predictor, det_px, (H, W))
        for cm, bb in cs:
            frac_in = (cm & boxcov).sum() / max(1, cm.sum())
            if frac_in >= COVER: detected += 1
            seg_dices.append(dice(sam_one(predictor, bb), cm))   # SAM cho ĐÚNG bbox mảnh
        # (F) box thừa: tâm trong specimen nhưng không trùng mảnh u nào (IoU bbox ~0)
        gtmask = gt
        fp = 0
        for b in det_px:
            cx = int(np.clip((b[0]+b[2])/2, 0, W-1)); cy = int(np.clip((b[1]+b[3])/2, 0, H-1))
            x0, y0, x1, y1 = [int(v) for v in b]
            reg = gtmask[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
            overlap = reg.mean() if reg.size else 0
            if overlap < 0.05:   # box gần như không chứa pixel u
                fp += 1
        rec = dict(stem=s, n_comp=len(cs), n_box=len(det_px),
                   recall=detected/len(cs), seg_ceil=float(np.mean(seg_dices)),
                   n_fp=fp, auto_dice=dice(auto_full, gt))
        rows.append(rec)
        print(f"  {s[:20]} comp={rec['n_comp']} box={rec['n_box']} "
              f"recall={rec['recall']:.2f} seg_ceil={rec['seg_ceil']:.2f} fp={fp} "
              f"auto={rec['auto_dice']:.2f}", flush=True)

    # ---- tổng hợp ----
    R = np.array([r["recall"] for r in rows]); S = np.array([r["seg_ceil"] for r in rows])
    A = np.array([r["auto_dice"] for r in rows]); FP = np.array([r["n_fp"] for r in rows])
    bad = A < 0.30
    print(f"\n===== PHÂN RÃ ĐA-U (N={len(rows)} ca đa-u) =====")
    print(f"RECALL mảnh (detector phủ): median={np.median(R):.2f} mean={np.mean(R):.2f} "
          f"| ca recall<1.0 (sót mảnh): {(R<0.999).sum()}/{len(rows)}")
    print(f"SEG-CEIL (SAM|box-tay-đúng-mảnh): median={np.median(S):.2f} mean={np.mean(S):.2f} "
          f"= TRẦN nếu localize hoàn hảo")
    print(f"BOX THỪA: tổng={int(FP.sum())} | ca có fp>=1: {(FP>=1).sum()}/{len(rows)}")
    print(f"\n-- nhóm AUTO<0.30 ({bad.sum()} ca) --")
    print(f"  recall median={np.median(R[bad]):.2f} | seg_ceil median={np.median(S[bad]):.2f} "
          f"| fp tổng={int(FP[bad].sum())}")
    # quy lỗi: với ca xấu, nếu recall thấp -> lỗi RECALL; nếu recall cao nhưng auto thấp & seg_ceil cao -> lỗi SEGMENT/box-lỏng
    print(f"\n-- quy lỗi từng ca xấu (auto<0.30) --")
    for r in sorted([x for x in rows if x["auto_dice"] < 0.30], key=lambda r: r["auto_dice"]):
        if r["recall"] < 0.6: tag = "RECALL (sót mảnh)"
        elif r["seg_ceil"] - r["auto_dice"] > 0.25: tag = "SEGMENT/box-lỏng (box phủ nhưng SAM sai)"
        else: tag = "khác"
        print(f"  {r['stem'][:20]} comp={r['n_comp']} recall={r['recall']:.2f} "
              f"seg_ceil={r['seg_ceil']:.2f} auto={r['auto_dice']:.2f} fp={r['n_fp']} -> {tag}")
    with open("results/handdraw_diag.csv", "w", newline="") as f:
        w = csv.writer(f); cols = ["stem", "n_comp", "n_box", "recall", "seg_ceil", "n_fp", "auto_dice"]
        w.writerow(cols)
        for r in rows: w.writerow([round(r[c], 4) if isinstance(r[c], float) else r[c] for c in cols])
    print("-> results/handdraw_diag.csv", flush=True)

if __name__ == "__main__":
    main()
