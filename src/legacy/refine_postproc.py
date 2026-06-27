"""
HƯỚNG MỚI (user 2026-06-26) — HẬU XỬ LÝ viền: coi mask của segmenter là KHỞI TẠO,
tinh chỉnh bằng Active Contours (Snakes) / edge-aware filter để bám cạnh ảnh gốc.
KHÔNG train. Áp lên champion segR_lab (ceiling 0.883 / full-auto ~0.635).

Phương pháp (pydensecrf build fail → dùng có sẵn):
  - guided : guided-filter (He et al.) làm mượt prob theo cấu trúc ảnh xám -> threshold (CRF-lite).
  - gac    : morphological_geodesic_active_contour (Snakes bám gradient) seed = mask.
  - acwe   : morphological_chan_vese (active contour vùng).
⚠️ Caveat: u & mô lành gần cùng màu (gốc của spill) -> cạnh yếu, refine có thể không cứu/hại.
Đo Dice TRƯỚC/SAU trên 50 vẽ tay (full-auto + ceiling), lưu results/refine_postproc.json.
Env: /home/hvusynh2/conda_envs/medsam2_anno/bin/python.
"""
import json, os, sys, csv, numpy as np, cv2, torch
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from skimage.segmentation import (morphological_geodesic_active_contour as MGAC,
                                  morphological_chan_vese as MCV, inverse_gaussian_gradient)
import predict_seg_crop as P
from seg_crop import pad_box, make_channels, SIZE, frag_boxes, ncomp
from detector import DenseDetector, propose_boxes
from specimen_clean import clean_specimen
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEVICE = "cuda"; AC = torch.autocast("cuda", dtype=torch.bfloat16)
CFG = "configs/sam2.1_hiera_t512"; CKPT = "checkpoints/sam2.1_hiera_tiny.pt"; RES = 1024
DET_CKPT = "checkpoints/detector_recall.pt"

def dice(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s

@torch.no_grad()
def soft_union(net, mode, bgr, boxes):
    """prob mềm full-res (max qua box)."""
    H, W = bgr.shape[:2]; prob = np.zeros((H, W), np.float32)
    for b in boxes:
        x0, y0, x1, y1 = pad_box(list(map(float, b)), 0.15, W, H)
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0: continue
        cr = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(make_channels(np.ascontiguousarray(cr), mode))[None].to(DEVICE)
        pm = torch.sigmoid(net(x))[0, 0].float().cpu().numpy()
        pm = cv2.resize(pm, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        prob[y0:y1, x0:x1] = np.maximum(prob[y0:y1, x0:x1], pm)
    return prob

def guided_filter(guide, src, r=8, eps=1e-3):
    """guided filter He et al. guide,src float [0,1] HxW -> output [0,1]."""
    g = guide.astype(np.float32); p = src.astype(np.float32)
    mean = lambda im: cv2.boxFilter(im, -1, (2 * r + 1, 2 * r + 1))
    mg, mp = mean(g), mean(p); mgp = mean(g * p); mgg = mean(g * g)
    cov = mgp - mg * mp; var = mgg - mg * mg
    a = cov / (var + eps); b = mp - a * mg
    return mean(a) * g + mean(b)

def refine(prob, bgr, method):
    """prob float HxW [0,1] -> mask bool sau refine. Chỉ refine trong vùng bao quanh mask."""
    binm = prob > 0.5
    if binm.sum() == 0: return binm
    ys, xs = np.where(binm); pad = 60
    y0 = max(0, ys.min() - pad); y1 = min(bgr.shape[0], ys.max() + pad)
    x0 = max(0, xs.min() - pad); x1 = min(bgr.shape[1], xs.max() + pad)
    sub = bgr[y0:y1, x0:x1]; sp = prob[y0:y1, x0:x1]; sb = binm[y0:y1, x0:x1]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    out = sb
    if method == "guided":
        ref = guided_filter(gray, sp, r=8, eps=1e-3)
        out = ref > 0.5
    elif method == "gac":
        gimg = inverse_gaussian_gradient(gray, alpha=100, sigma=4)
        out = MGAC(gimg, num_iter=12, init_level_set=sb.astype(np.uint8),
                   smoothing=2, balloon=-1) > 0  # balloon âm = co lại
    elif method == "acwe":
        out = MCV(gray, num_iter=12, init_level_set=sb.astype(np.uint8),
                  smoothing=2, lambda1=1, lambda2=1) > 0
    full = np.zeros(binm.shape, bool); full[y0:y1, x0:x1] = out
    return full

def main():
    model = build_sam2(CFG, CKPT, device=DEVICE, hydra_overrides_extra=[f"++model.image_size={RES}"])
    predictor = SAM2ImagePredictor(model)
    dk = torch.load(DET_CKPT, weights_only=False)
    det = DenseDetector(grid=dk.get("grid", 64)).to(DEVICE); det.load_state_dict(dk["det"]); det.eval()
    net, mode, _ = P.load_segmenter()
    print(f"segmenter={P.SEG_CKPT} ({mode}) | refine methods: none/guided/gac/acwe", flush=True)
    stems = json.load(open("labels_handdraw/select.json"))["stems"]
    have = [s for s in stems if os.path.isfile(f"labels_handdraw/masks/{s}.png")]
    methods = ["none", "guided", "gac", "acwe"]
    res = {m: {"auto": [], "ceil": []} for m in methods}
    grp = []
    for s in have:
        gt = cv2.imread(f"labels_handdraw/masks/{s}.png", 0) > 127
        bgr = cv2.imread(f"data/20241212/{s}.jpg"); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = gt.shape; grp.append(ncomp(gt))
        with torch.inference_mode(), AC:
            predictor.set_image(rgb); feat = predictor._features["image_embed"].float()
            obj, boxes = det(feat)
        spec, _, _ = clean_specimen(bgr)
        ab = propose_boxes(obj[0].float(), boxes[0].float(), H, W, spec=spec, thr=0.5)
        gb = frag_boxes(gt)
        pa = soft_union(net, mode, bgr, ab) if len(ab) else np.zeros((H, W), np.float32)
        pc = soft_union(net, mode, bgr, gb) if gb else np.zeros((H, W), np.float32)
        for m in methods:
            ma = (pa > 0.5) if m == "none" else refine(pa, bgr, m)
            mc = (pc > 0.5) if m == "none" else refine(pc, bgr, m)
            res[m]["auto"].append(dice(ma, gt)); res[m]["ceil"].append(dice(mc, gt))
        print(f"  {s[:20]:22} none auto={res['none']['auto'][-1]:.3f} ceil={res['none']['ceil'][-1]:.3f} | "
              f"guided ceil={res['guided']['ceil'][-1]:.3f} gac ceil={res['gac']['ceil'][-1]:.3f}", flush=True)
    import statistics as st
    grp = np.array(grp)
    print(f"\n===== HẬU XỬ LÝ refine trên {len(have)} vẽ tay (so champion segR_lab) =====")
    out = {}
    for m in methods:
        a = res[m]["auto"]; c = res[m]["ceil"]
        am, cm = st.median(a), st.median(c)
        a1 = st.median([a[i] for i in range(len(a)) if grp[i] <= 1]); am1 = st.median([a[i] for i in range(len(a)) if grp[i] > 1])
        out[m] = {"auto_median": am, "ceil_median": cm, "auto_mean": st.mean(a), "ceil_mean": st.mean(c)}
        print(f"  {m:7} full-auto median={am:.4f} mean={st.mean(a):.4f} | ceiling median={cm:.4f} mean={st.mean(c):.4f}")
    print("SO: none = champion (auto~0.635 / ceil 0.883). >0 = refine giúp.", flush=True)
    json.dump(out, open("results/refine_postproc.json", "w"), indent=1)
    print("-> results/refine_postproc.json", flush=True)

if __name__ == "__main__":
    main()
