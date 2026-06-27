"""
CHẨN ĐOÁN + THỬ NGHIỆM trên test VẼ TAY:
 (A) Đếm box detector rơi NGOÀI bệnh phẩm (= thước/nhãn/nền) -> trả lời "còn bắt thước/nhãn?"
 (B) So Dice các cấu hình: baseline(thr,no-gate) | +specimen-gate | +gate & hạ ngưỡng (recall đa-u).
set_image 1 lần / ảnh, tái dùng feature cho detector + tất cả cấu hình -> tiết kiệm GPU.
Chạy nền: .../python -u scratch_handdraw_gate.py > results/handdraw_gate.log 2>&1 &
"""
import os, sys, json, csv, numpy as np, cv2, torch
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from detector import DenseDetector, decode_detections, cxcywh_to_xyxy
from specimen_clean import clean_specimen
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CFG = "configs/sam2.1_hiera_t512"; CKPT = "checkpoints/sam2.1_hiera_tiny.pt"
DET_CKPT = "checkpoints/detector_maskloss.pt"
RES = 1024; AC = torch.autocast("cuda", dtype=torch.bfloat16)
HMASK = "labels_handdraw/masks"
MIN_FRAC = 0.002
# Cấu hình: (tên, thr, gate?)
CONFIGS = [("base@.5", 0.5, False), ("gate@.5", 0.5, True),
           ("gate@.35", 0.35, True), ("gate@.25", 0.25, True), ("gate@.15", 0.15, True)]

def dice(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s
def ncomp(m):
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return max(1, int((st[1:, cv2.CC_STAT_AREA] >= MIN_FRAC * m.size).sum()))

def detector_raw(det, predictor, rgb, thr):
    """Trả boxes_px (n,4) xyxy ở thr cho trước (dùng feature đã set_image)."""
    H, W = rgb.shape[:2]
    feat = predictor._features["image_embed"].float()
    with torch.inference_mode(), AC:
        obj, boxes = det(feat)
    bxk, _ = decode_detections(obj[0].float(), boxes[0].float(), thr=thr)
    xyxy = cxcywh_to_xyxy(bxk).clamp(0, 1).cpu().numpy()
    return (xyxy * np.array([W, H, W, H], np.float32)).astype(np.float32)

def gate_boxes(boxes_px, spec):
    """Giữ box có TÂM nằm trong bệnh phẩm (spec mask). spec rỗng -> giữ hết."""
    if spec is None or spec.sum() == 0:
        return boxes_px, np.zeros(len(boxes_px), bool)
    H, W = spec.shape
    keep = np.ones(len(boxes_px), bool)
    for i, b in enumerate(boxes_px):
        cx = int(np.clip((b[0] + b[2]) / 2, 0, W - 1)); cy = int(np.clip((b[1] + b[3]) / 2, 0, H - 1))
        keep[i] = spec[cy, cx] > 0
    return boxes_px[keep], ~keep   # (kept boxes, dropped_mask)

def sam_union(predictor, rgb, boxes_px):
    H, W = rgb.shape[:2]; union = np.zeros((H, W), bool)
    if len(boxes_px) == 0: return union
    with torch.inference_mode(), AC:
        for b in boxes_px:
            mk, sc, _ = predictor.predict(box=b, multimask_output=True)
            union |= mk[int(np.argmax(sc))].astype(bool)
    return union

def main():
    stems = json.load(open("labels_handdraw/select.json"))["stems"]
    have = [s for s in stems if os.path.isfile(f"{HMASK}/{s}.png")]
    model = build_sam2(CFG, CKPT, device="cuda", hydra_overrides_extra=[f"++model.image_size={RES}"])
    predictor = SAM2ImagePredictor(model)
    ck = torch.load(DET_CKPT, weights_only=False)
    det = DenseDetector().to("cuda"); det.load_state_dict(ck["det"]); det.eval()
    print(f"detector={DET_CKPT} ep{ck['epoch']} | N={len(have)}", flush=True)

    rows = []; n_drop_total = 0; n_box_total = 0; imgs_with_drop = 0
    for s in have:
        gt = cv2.imread(f"{HMASK}/{s}.png", 0) > 127
        bgr = cv2.imread(f"data/20241212/{s}.jpg"); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        no = ncomp(gt)
        spec, _, sdbg = clean_specimen(bgr)
        with torch.inference_mode(), AC:
            predictor.set_image(rgb)     # ENCODER 1 lần -> feature dùng chung
        rec = {"stem": s, "n_obj": no, "n_ruler": len(sdbg["ruler"]), "n_label": len(sdbg["label"])}
        # (A) chẩn đoán thước/nhãn ở thr=0.5
        b50 = detector_raw(det, predictor, rgb, 0.5)
        _, dropped = gate_boxes(b50, spec)
        rec["n_box@.5"] = len(b50); rec["n_drop@.5"] = int(dropped.sum())
        n_box_total += len(b50); n_drop_total += int(dropped.sum())
        if dropped.sum() > 0: imgs_with_drop += 1
        # (B) Dice mỗi cấu hình
        for name, thr, gate in CONFIGS:
            bx = detector_raw(det, predictor, rgb, thr)
            if gate: bx, _ = gate_boxes(bx, spec)
            mk = sam_union(predictor, rgb, bx)
            rec[name] = dice(mk, gt); rec[f"nb_{name}"] = len(bx)
        rows.append(rec)
        print(f"  {s[:20]} n_u={no} box.5={rec['n_box@.5']} drop={rec['n_drop@.5']} | "
              + " ".join(f"{n}={rec[n]:.2f}" for n, _, _ in CONFIGS), flush=True)

    # ---- tổng hợp ----
    one = lambda r: r["n_obj"] <= 1; mul = lambda r: r["n_obj"] > 1
    def med(key, f=None):
        v = np.array([r[key] for r in rows if (f is None or f(r))], float); v = v[~np.isnan(v)]
        return (np.median(v), np.mean(v), len(v))
    print(f"\n===== (A) THƯỚC/NHÃN ngoài bệnh phẩm @thr=0.5 =====")
    print(f"Tổng box={n_box_total} | rơi-ngoài-bệnh-phẩm={n_drop_total} "
          f"({100*n_drop_total/max(1,n_box_total):.1f}%) | ảnh dính={imgs_with_drop}/{len(rows)}")
    print(f"\n===== (B) Dice theo cấu hình (median | 1u | >1u) =====")
    for name, _, _ in CONFIGS:
        a = med(name); o = med(name, one); m = med(name, mul)
        print(f"  {name:10s} median={a[0]:.4f} mean={a[1]:.4f} | 1u={o[0]:.3f} >1u={m[0]:.3f}")
    cols = ["stem", "n_obj", "n_ruler", "n_label", "n_box@.5", "n_drop@.5"] + \
           [c for n, _, _ in CONFIGS for c in (n, f"nb_{n}")]
    with open("results/handdraw_gate.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows: w.writerow([round(r[c], 4) if isinstance(r[c], float) else r.get(c, "") for c in cols])
    print("-> results/handdraw_gate.csv", flush=True)

if __name__ == "__main__":
    main()
