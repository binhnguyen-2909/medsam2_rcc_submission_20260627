"""
ELLIPSE Ở KỊCH BẢN TỐT NHẤT (red-ceiling) — đo TRẦN của phương án sinh hình elip.
Thay vì fit ellipse trên mask segmenter (chưa hoàn hảo), fit trên:
  (1) GT trực tiếp        -> ellipse_of_GT  = TRẦN TUYỆT ĐỐI của ellipse (u có đủ 'elip' không?)
  (2) red-ceiling mask    = GT ∩ union(box) (segment hoàn hảo trong box) -> rồi fit ellipse
      cho cả box detector (auto) và box GT-mảnh (ceil).
So với: red-ceiling thường (không ellipse) ~0.938; ceiling segmenter 0.883; full-auto 0.635.
Chạy: python blob_redceiling.py -> results/blob_redceiling.json. Env: medsam2_anno.
"""
import json, os, sys, numpy as np, cv2, torch, statistics as st
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from seg_crop import frag_boxes, ncomp
from detector import DenseDetector, propose_boxes
from specimen_clean import clean_specimen
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
AC = torch.autocast("cuda", dtype=torch.bfloat16); DEVICE = "cuda"; MINF = 0.002

def dice(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s

def boxes_union_mask(boxes, H, W):
    u = np.zeros((H, W), bool)
    for b in boxes:
        x0, y0, x1, y1 = [int(max(0, v)) for v in b]
        x1 = min(W, x1); y1 = min(H, y1)
        u[y0:y1, x0:x1] = True
    return u

def ellipse_fill(mask):
    """fit + tô đầy ellipse mỗi connected-comp >= MINF -> mask làm tròn."""
    m = mask.astype(np.uint8); H, W = mask.shape
    n, lab, st_, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros((H, W), np.uint8); thr = MINF * m.size
    for i in range(1, n):
        if st_[i, cv2.CC_STAT_AREA] < thr: continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        c = max(cnts, key=cv2.contourArea)
        if len(c) < 5:
            cv2.drawContours(out, [c], -1, 1, -1); continue
        cv2.ellipse(out, cv2.fitEllipse(c), 1, -1)
    return out.astype(bool)

def main():
    model = build_sam2("configs/sam2.1_hiera_t512", "checkpoints/sam2.1_hiera_tiny.pt",
                       device=DEVICE, hydra_overrides_extra=["++model.image_size=1024"])
    pred = SAM2ImagePredictor(model)
    dk = torch.load("checkpoints/detector_recall.pt", weights_only=False)
    det = DenseDetector(grid=dk.get("grid", 64)).to(DEVICE); det.load_state_dict(dk["det"]); det.eval()
    stems = json.load(open("labels_handdraw/select.json"))["stems"]
    have = [s for s in stems if os.path.isfile(f"labels_handdraw/masks/{s}.png")]
    keys = ["ellipse_of_GT", "auto_redceil", "auto_redceil_ell", "ceil_redceil", "ceil_redceil_ell"]
    res = {k: [] for k in keys}; grp = []
    for s in have:
        gt = cv2.imread(f"labels_handdraw/masks/{s}.png", 0) > 127
        bgr = cv2.imread(f"data/20241212/{s}.jpg"); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = gt.shape; grp.append(ncomp(gt)); spec, _, _ = clean_specimen(bgr)
        with torch.inference_mode(), AC:
            pred.set_image(rgb); feat = pred._features["image_embed"].float(); obj, bx = det(feat)
        ab = list(propose_boxes(obj[0].float(), bx[0].float(), H, W, spec=spec, thr=0.5))
        gb = frag_boxes(gt)
        # (1) ellipse trực tiếp từ GT
        res["ellipse_of_GT"].append(dice(ellipse_fill(gt), gt))
        # (2) red-ceiling = GT ∩ union(box), rồi ellipse
        for src, boxes in (("auto", ab), ("ceil", gb)):
            rc = gt & boxes_union_mask(boxes, H, W)
            res[f"{src}_redceil"].append(dice(rc, gt))
            res[f"{src}_redceil_ell"].append(dice(ellipse_fill(rc), gt))
        print(f"  {s[:20]:22} ellGT={res['ellipse_of_GT'][-1]:.3f} | auto rc={res['auto_redceil'][-1]:.3f} "
              f"rc_ell={res['auto_redceil_ell'][-1]:.3f} | ceil rc={res['ceil_redceil'][-1]:.3f} "
              f"rc_ell={res['ceil_redceil_ell'][-1]:.3f}", flush=True)
    grp = np.array(grp)
    def agg(k, mul=None):
        v = res[k]
        idx = range(len(v)) if mul is None else ([i for i in range(len(v)) if (grp[i] > 1) == mul])
        vv = [v[i] for i in idx]; return st.median(vv), st.mean(vv)
    print("\n===== ELLIPSE Ở RED-CEILING (50 vẽ tay) =====")
    out = {}
    for k in keys:
        md, mn = agg(k); o = agg(k, False); m = agg(k, True)
        out[k] = {"median": md, "mean": mn, "1u": o[0], ">1u": m[0]}
        print(f"  {k:18} median={md:.4f} mean={mn:.4f} | 1u={o[0]:.3f} >1u={m[0]:.3f}")
    print("\nSO: red-ceiling thường ~0.938 | ceiling segmenter 0.883 | full-auto 0.635")
    print("=> ellipse_of_GT = trần tuyệt đối của hình elip (u có đủ 'elip' không)")
    json.dump(out, open("results/blob_redceiling.json", "w"), indent=1)
    print("-> results/blob_redceiling.json", flush=True)

if __name__ == "__main__":
    main()
