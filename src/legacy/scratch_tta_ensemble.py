"""
THÍ NGHIỆM MIỄN PHÍ TĂNG RECALL (không train lại, không cần nhãn mới):
  Flip-TTA + ensemble nhiều checkpoint -> gộp box (NMS) -> SAM union -> Dice vs mask vẽ tay.
So 4 cấu hình trên 50 ảnh vẽ tay. Mục tiêu: vớt mảnh u bị sót (nút thắt localize).
Chạy: /home/hvusynh2/conda_envs/medsam2_anno/bin/python -u scratch_tta_ensemble.py
"""
import os, sys, json, csv, numpy as np, cv2, torch
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from detector import DenseDetector, propose_boxes, pairwise_iou, cxcywh_to_xyxy
from specimen_clean import clean_specimen
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CFG = "configs/sam2.1_hiera_t512"; CKPT = "checkpoints/sam2.1_hiera_tiny.pt"
RES = 1024; AC = torch.autocast("cuda", dtype=torch.bfloat16)
HMASK = "labels_handdraw/masks"; DET_THR = 0.5
CKPTS = {"recall": "checkpoints/detector_recall.pt",
         "maskloss": "checkpoints/detector_maskloss.pt",
         "ms": "checkpoints/detector_recall_ms.pt"}

def dice(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s

def nms_merge(boxes, iou_thr=0.7):
    """gộp list box pixel xyxy, bỏ box trùng (IoU>thr), giữ box LỚN trước."""
    if len(boxes) == 0: return np.zeros((0, 4), np.float32)
    b = np.array(boxes, np.float32)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    order = np.argsort(-area); t = torch.as_tensor(b)
    kept = []
    for i in order:
        ok = True
        for j in kept:
            iou = float(pairwise_iou(t[int(i):int(i)+1], t[int(j):int(j)+1])[0, 0])
            if iou > iou_thr: ok = False; break
        if ok: kept.append(int(i))
    return b[kept]

def sam_union(predictor_feats_set, rgb, boxes):
    H, W = rgb.shape[:2]; union = np.zeros((H, W), bool)
    if len(boxes) == 0: return union
    with torch.inference_mode(), AC:
        for box in boxes:
            mk, sc, _ = predictor_feats_set.predict(box=np.array(box, np.float32), multimask_output=True)
            union |= mk[int(np.argmax(sc))].astype(bool)
    return union

def det_boxes(det, predictor, rgb, spec, thr):
    """forward detector trên rgb (đã set_image trước) -> box pixel xyxy (theo frame rgb)."""
    H, W = rgb.shape[:2]
    feat = predictor._features["image_embed"].float()
    with torch.inference_mode(), AC:
        obj, boxes = det(feat)
    return propose_boxes(obj[0].float(), boxes[0].float(), H, W, spec=spec, thr=thr)

def main():
    stems = json.load(open("labels_handdraw/select.json"))["stems"]
    have = [s for s in stems if os.path.isfile(f"{HMASK}/{s}.png")]
    print(f"N={len(have)}", flush=True)
    model = build_sam2(CFG, CKPT, device="cuda", hydra_overrides_extra=[f"++model.image_size={RES}"])
    predictor = SAM2ImagePredictor(model)
    dets = {}
    for k, p in CKPTS.items():
        ck = torch.load(p, weights_only=False)
        d = DenseDetector(grid=ck.get("grid", 64) or 64).to("cuda"); d.load_state_dict(ck["det"]); d.eval()
        dets[k] = d
        print(f"  {k}: ep{ck['epoch']} grid{ck.get('grid')}", flush=True)

    CONFIGS = ["recall_orig", "recall_tta", "ens3_orig", "ens3_tta"]
    res = {c: [] for c in CONFIGS}; nbox = {c: [] for c in CONFIGS}
    nobj = []
    for s in have:
        gt = cv2.imread(f"{HMASK}/{s}.png", 0) > 127
        bgr = cv2.imread(f"data/20241212/{s}.jpg"); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        spec, _, _ = clean_specimen(bgr)
        spec_f = spec[:, ::-1].copy() if spec is not None else None
        rgb_f = rgb[:, ::-1].copy()
        # đếm mảnh GT để tách 1u/>1u
        nlab, _, st, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
        nobj.append(max(1, int((st[1:, cv2.CC_STAT_AREA] >= 0.002 * gt.size).sum())))

        # set_image gốc -> box mọi ckpt; set_image lật -> box mọi ckpt (map về frame gốc)
        with torch.inference_mode(), AC:
            predictor.set_image(rgb)
        bx_orig = {k: det_boxes(dets[k], predictor, rgb, spec, DET_THR) for k in dets}
        # SAM trên features gốc cho mọi config (predictor đang giữ features gốc)
        def flip_back(b):
            if len(b) == 0: return b
            b = b.copy(); x0 = W - b[:, 2]; x1 = W - b[:, 0]; b[:, 0] = x0; b[:, 2] = x1; return b
        with torch.inference_mode(), AC:
            predictor.set_image(rgb_f)
        bx_flip = {k: flip_back(det_boxes(dets[k], predictor, rgb_f, spec_f, DET_THR)) for k in dets}
        # set lại features gốc cho SAM decode
        with torch.inference_mode(), AC:
            predictor.set_image(rgb)

        cfg_boxes = {
            "recall_orig": bx_orig["recall"],
            "recall_tta":  nms_merge([*bx_orig["recall"], *bx_flip["recall"]]),
            "ens3_orig":   nms_merge([*bx_orig["recall"], *bx_orig["maskloss"], *bx_orig["ms"]]),
            "ens3_tta":    nms_merge([*bx_orig["recall"], *bx_flip["recall"],
                                      *bx_orig["maskloss"], *bx_flip["maskloss"],
                                      *bx_orig["ms"], *bx_flip["ms"]]),
        }
        line = f"  {s[:20]} n_u={nobj[-1]} |"
        for c in CONFIGS:
            bxs = cfg_boxes[c]
            mk = sam_union(predictor, rgb, bxs)
            d = dice(mk, gt); res[c].append(d); nbox[c].append(len(bxs))
            line += f" {c.split('_')[1] if c!='recall_orig' else 'base'}={d:.2f}"
        print(line, flush=True)

    nobj = np.array(nobj); one = nobj <= 1; mul = nobj > 1
    print(f"\n===== TTA/ENSEMBLE (N={len(have)}: {one.sum()} 1u + {mul.sum()} >1u) =====")
    print(f"{'config':14s} {'median':>7s} {'mean':>7s} {'1u':>6s} {'>1u':>6s} {'#box':>6s}")
    rows = []
    for c in CONFIGS:
        v = np.array(res[c])
        md, mn = np.median(v), np.mean(v)
        o = np.median(v[one]) if one.any() else 0; m = np.median(v[mul]) if mul.any() else 0
        nb = np.mean(nbox[c])
        print(f"{c:14s} {md:7.3f} {mn:7.3f} {o:6.3f} {m:6.3f} {nb:6.1f}")
        rows.append([c, round(md, 4), round(mn, 4), round(o, 4), round(m, 4), round(nb, 2)])
    with open("results/tta_ensemble.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config", "median", "mean", "med_1u", "med_mul", "avg_box"]); w.writerows(rows)
    print("-> results/tta_ensemble.csv", flush=True)

if __name__ == "__main__":
    main()
