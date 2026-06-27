"""
LOCALIZE #E (user) — GỘP BOX bằng blob detection để giảm "nhiều box cùng 1 chỗ -> cộng dồn spill".
detector nhả nhiều box chồng lên 1 u -> mỗi box segment lệch 1 kiểu -> union cộng dồn lỗi.
Ý: rasterize tất cả box -> blob (connected-comp) HỢP NHẤT box trùng/chạm nhau -> 1 box/cụm
-> segment mỗi cụm 1 lần. Biến thể: dilate trước khi gộp (gộp box gần nhau).
Chạy: python blob_merge.py -> results/blob_merge.json. Env: medsam2_anno.
"""
import json, os, sys, numpy as np, cv2, torch, statistics as st
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import predict_seg_crop as P
from seg_crop import frag_boxes, ncomp
from detector import DenseDetector, propose_boxes
from specimen_clean import clean_specimen
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
AC = torch.autocast("cuda", dtype=torch.bfloat16); DEVICE = "cuda"

def dice(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s

def merge_boxes(boxes, H, W, dilate=0):
    """rasterize box -> blob CC -> bbox mỗi cụm (gộp box chồng/chạm). dilate>0 gộp box gần nhau."""
    if not boxes: return []
    bm = np.zeros((H, W), np.uint8)
    for b in boxes:
        x0, y0, x1, y1 = [int(max(0, v)) for v in b]
        cv2.rectangle(bm, (x0, y0), (min(W - 1, x1), min(H - 1, y1)), 1, -1)
    if dilate > 0:
        k = np.ones((dilate, dilate), np.uint8); bm = cv2.dilate(bm, k)
    n, lab, st_, _ = cv2.connectedComponentsWithStats(bm, 8); out = []
    for i in range(1, n):
        x0, y0 = st_[i, cv2.CC_STAT_LEFT], st_[i, cv2.CC_STAT_TOP]
        out.append([float(x0), float(y0), float(x0 + st_[i, cv2.CC_STAT_WIDTH]), float(y0 + st_[i, cv2.CC_STAT_HEIGHT])])
    return out

def seg(net, mode, bgr, boxes):
    return P.boxes_to_mask(net, mode, DEVICE, bgr, boxes) if boxes else np.zeros(bgr.shape[:2], bool)

def main():
    model = build_sam2("configs/sam2.1_hiera_t512", "checkpoints/sam2.1_hiera_tiny.pt",
                       device=DEVICE, hydra_overrides_extra=["++model.image_size=1024"])
    pred = SAM2ImagePredictor(model)
    dk = torch.load("checkpoints/detector_recall.pt", weights_only=False)
    det = DenseDetector(grid=dk.get("grid", 64)).to(DEVICE); det.load_state_dict(dk["det"]); det.eval()
    net, mode, _ = P.load_segmenter()
    stems = json.load(open("labels_handdraw/select.json"))["stems"]
    have = [s for s in stems if os.path.isfile(f"labels_handdraw/masks/{s}.png")]
    variants = ["base", "merge", "merge_d40"]
    res = {v: [] for v in variants}; grp = []; nbox = {"raw": [], "merge": []}
    for s in have:
        gt = cv2.imread(f"labels_handdraw/masks/{s}.png", 0) > 127
        bgr = cv2.imread(f"data/20241212/{s}.jpg"); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = gt.shape; grp.append(ncomp(gt)); spec, _, _ = clean_specimen(bgr)
        with torch.inference_mode(), AC:
            pred.set_image(rgb); feat = pred._features["image_embed"].float(); obj, bx = det(feat)
        ab = list(propose_boxes(obj[0].float(), bx[0].float(), H, W, spec=spec, thr=0.5))
        mb = merge_boxes(ab, H, W, 0); mbd = merge_boxes(ab, H, W, 40)
        nbox["raw"].append(len(ab)); nbox["merge"].append(len(mb))
        res["base"].append(dice(seg(net, mode, bgr, ab), gt))
        res["merge"].append(dice(seg(net, mode, bgr, mb), gt))
        res["merge_d40"].append(dice(seg(net, mode, bgr, mbd), gt))
        print(f"  {s[:20]:22} nbox {len(ab)}->{len(mb)} | base={res['base'][-1]:.3f} "
              f"merge={res['merge'][-1]:.3f} merge_d40={res['merge_d40'][-1]:.3f}", flush=True)
    grp = np.array(grp)
    def agg(v, mul=None):
        x = res[v]; idx = range(len(x)) if mul is None else [i for i in range(len(x)) if (grp[i] > 1) == mul]
        xx = [x[i] for i in idx]; return st.median(xx), st.mean(xx)
    print(f"\n===== GỘP BOX (blob) trên 50 vẽ tay =====")
    print(f"#box trung vị: raw={int(np.median(nbox['raw']))} -> merge={int(np.median(nbox['merge']))} "
          f"(mean {np.mean(nbox['raw']):.1f}->{np.mean(nbox['merge']):.1f})")
    out = {}
    for v in variants:
        a = agg(v); o = agg(v, False); m = agg(v, True)
        out[v] = {"median": a[0], "mean": a[1], "1u": o[0], ">1u": m[0]}
        print(f"  {v:10} median={a[0]:.4f} mean={a[1]:.4f} | 1u={o[0]:.3f} >1u={m[0]:.3f}")
    print("SO: base = detector 0.635 | ceiling 0.883")
    json.dump(out, open("results/blob_merge.json", "w"), indent=1)
    print("-> results/blob_merge.json", flush=True)

if __name__ == "__main__":
    main()
