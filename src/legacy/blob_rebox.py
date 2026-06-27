"""
LOCALIZE #C — REFINE HÌNH HỌC từ mask đã tô: segmenter -> mask -> blob detection / fit ellipse
(cv2) -> sinh BOX/ELLIPSE mới -> (a) segment lại từ box mới, hoặc (b) dùng ellipse làm mask cuối
(làm tròn, cắt 'tua' spill vì u gần hình elip). Xem box/mask mới có ĐÚNG hơn không.

Biến thể (đo trên 50 vẽ tay, cả FULL-AUTO box-detector lẫn CEILING box-GT):
  base         : detector/GT box -> seg (mốc 0.635 / 0.883)
  cc_rebox     : seg -> bbox mỗi connected-comp -> seg lại -> union
  ell_rebox    : seg -> fitEllipse -> bbox ellipse -> seg lại -> union
  ell_fill     : seg -> fitEllipse -> tô đầy ellipse = mask cuối (KHÔNG seg lại)
  cc_rebox x2  : lặp cc_rebox 2 lần
Chạy: python blob_rebox.py  -> results/blob_rebox.json. Env: medsam2_anno.
"""
import json, os, sys, numpy as np, cv2, torch, statistics as st
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import predict_seg_crop as P
from seg_crop import frag_boxes, ncomp, pad_box, make_channels, SIZE
from detector import DenseDetector, propose_boxes
from specimen_clean import clean_specimen
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
AC = torch.autocast("cuda", dtype=torch.bfloat16); DEVICE = "cuda"; MINF = 0.002

def dice(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s

@torch.no_grad()
def seg_boxes(net, mode, bgr, boxes):
    return P.boxes_to_mask(net, mode, DEVICE, bgr, boxes) if boxes else np.zeros(bgr.shape[:2], bool)

def comp_boxes(mask):
    """bbox mỗi connected component >= MINF."""
    m = mask.astype(np.uint8); n, lab, st_, _ = cv2.connectedComponentsWithStats(m, 8)
    thr = MINF * m.size; out = []
    for i in range(1, n):
        if st_[i, cv2.CC_STAT_AREA] < thr: continue
        x, y = st_[i, cv2.CC_STAT_LEFT], st_[i, cv2.CC_STAT_TOP]
        out.append([float(x), float(y), float(x + st_[i, cv2.CC_STAT_WIDTH]), float(y + st_[i, cv2.CC_STAT_HEIGHT])])
    return out

def ellipses(mask):
    """fitEllipse mỗi comp -> list (ellipse, bbox_of_ellipse). Cần >=5 điểm contour."""
    m = mask.astype(np.uint8); n, lab, st_, _ = cv2.connectedComponentsWithStats(m, 8)
    thr = MINF * m.size; out = []
    for i in range(1, n):
        if st_[i, cv2.CC_STAT_AREA] < thr: continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        c = max(cnts, key=cv2.contourArea)
        if len(c) < 5: continue
        e = cv2.fitEllipse(c)
        (cx, cy), (MA, ma), ang = e
        rx = max(MA, ma) / 2
        bb = [cx - rx, cy - rx, cx + rx, cy + rx]  # bbox vuông bao ellipse (an toàn)
        out.append((e, bb))
    return out

def ellipse_fill(mask):
    """tô đầy ellipse cho mỗi comp -> mask làm tròn."""
    H, W = mask.shape; out = np.zeros((H, W), np.uint8)
    for e, _ in ellipses(mask):
        cv2.ellipse(out, e, 1, -1)
    return out.astype(bool)

def main():
    model = build_sam2("configs/sam2.1_hiera_t512", "checkpoints/sam2.1_hiera_tiny.pt",
                       device=DEVICE, hydra_overrides_extra=["++model.image_size=1024"])
    pred = SAM2ImagePredictor(model)
    dk = torch.load("checkpoints/detector_recall.pt", weights_only=False)
    det = DenseDetector(grid=dk.get("grid", 64)).to(DEVICE); det.load_state_dict(dk["det"]); det.eval()
    net, mode, _ = P.load_segmenter()
    stems = json.load(open("labels_handdraw/select.json"))["stems"]
    have = [s for s in stems if os.path.isfile(f"labels_handdraw/masks/{s}.png")]
    variants = ["base", "cc_rebox", "ell_rebox", "ell_fill", "cc_rebox2"]
    # cho cả full-auto (box detector) và ceiling (box GT)
    res = {f"{src}_{v}": [] for src in ("auto", "ceil") for v in variants}
    grp = []
    for s in have:
        gt = cv2.imread(f"labels_handdraw/masks/{s}.png", 0) > 127
        bgr = cv2.imread(f"data/20241212/{s}.jpg"); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = gt.shape; grp.append(ncomp(gt))
        spec, _, _ = clean_specimen(bgr)
        with torch.inference_mode(), AC:
            pred.set_image(rgb); feat = pred._features["image_embed"].float(); obj, bx = det(feat)
        ab = list(propose_boxes(obj[0].float(), bx[0].float(), H, W, spec=spec, thr=0.5))
        gb = frag_boxes(gt)
        for src, boxes0 in (("auto", ab), ("ceil", gb)):
            m0 = seg_boxes(net, mode, bgr, boxes0)
            res[f"{src}_base"].append(dice(m0, gt))
            # cc_rebox
            m1 = seg_boxes(net, mode, bgr, comp_boxes(m0)) if m0.any() else m0
            res[f"{src}_cc_rebox"].append(dice(m1, gt))
            # cc_rebox x2
            m2 = seg_boxes(net, mode, bgr, comp_boxes(m1)) if m1.any() else m1
            res[f"{src}_cc_rebox2"].append(dice(m2, gt))
            # ell_rebox
            ebx = [bb for _, bb in ellipses(m0)]
            mr = seg_boxes(net, mode, bgr, ebx) if ebx else m0
            res[f"{src}_ell_rebox"].append(dice(mr, gt))
            # ell_fill (làm tròn, không seg lại)
            res[f"{src}_ell_fill"].append(dice(ellipse_fill(m0), gt))
        print(f"  {s[:20]:22} auto base={res['auto_base'][-1]:.3f} cc={res['auto_cc_rebox'][-1]:.3f} "
              f"ellfill={res['auto_ell_fill'][-1]:.3f} | ceil base={res['ceil_base'][-1]:.3f} "
              f"ellfill={res['ceil_ell_fill'][-1]:.3f}", flush=True)
    grp = np.array(grp)
    def agg(key, mul=None):
        v = res[key]
        if mul is None: idx = range(len(v))
        elif mul: idx = [i for i in range(len(v)) if grp[i] > 1]
        else: idx = [i for i in range(len(v)) if grp[i] <= 1]
        vv = [v[i] for i in idx]; return st.median(vv), st.mean(vv)
    print("\n===== REFINE HÌNH HỌC (blob/ellipse) trên 50 vẽ tay =====")
    out = {}
    for src, mark in (("auto", "FULL-AUTO (mốc 0.635)"), ("ceil", "CEILING (mốc 0.883)")):
        print(f"\n[{mark}]")
        for v in variants:
            k = f"{src}_{v}"; md, mn = agg(k); o = agg(k, False); m = agg(k, True)
            out[k] = {"median": md, "mean": mn, "1u": o[0], ">1u": m[0]}
            print(f"  {v:11} median={md:.4f} mean={mn:.4f} | 1u={o[0]:.3f} >1u={m[0]:.3f}")
    json.dump(out, open("results/blob_rebox.json", "w"), indent=1)
    print("\n-> results/blob_rebox.json", flush=True)

if __name__ == "__main__":
    main()
