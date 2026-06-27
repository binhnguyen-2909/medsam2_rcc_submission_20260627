"""
#F TENT — Test-Time Entropy Minimization trên segmenter SegResNet+LAB (champion).
Mỗi crop test: forward -> entropy(prob) -> backprop cập nhật affine của GroupNorm (Tent gốc
cập nhật BN; ở đây GroupNorm) K vòng -> rồi predict. Reset trọng số sau mỗi ảnh.
Eval ceiling (box GT) + full-auto trên 50 vẽ tay. So 0.883 / 0.635.
  python tent_eval.py --iters 10 --lr 1e-3
Env: medsam2_anno.
"""
import argparse, json, os, sys, copy, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F, statistics as st
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import predict_seg_crop as P
from seg_crop import frag_boxes, ncomp, pad_box, make_channels, SIZE, build_model, n_channels
from detector import DenseDetector, propose_boxes
from specimen_clean import clean_specimen
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
AC = torch.autocast("cuda", dtype=torch.bfloat16); DEVICE = "cuda"

def dice(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s

def ent(logit):
    p = torch.sigmoid(logit); p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3); args = ap.parse_args()
    ck = torch.load("checkpoints/seg_crop_segR_lab.pt", weights_only=False)
    mode = ck["channels"]; inC = n_channels(mode)
    net = build_model(ck["arch"], inC).to(DEVICE); net.load_state_dict(ck["net"])
    orig = copy.deepcopy(net.state_dict())
    # tham số GroupNorm affine để adapt
    gn_params = [p for m in net.modules() if isinstance(m, nn.GroupNorm) for p in (m.weight, m.bias) if p is not None]
    print(f"[tent] GroupNorm affine params={len(gn_params)} | iters={args.iters} lr={args.lr}", flush=True)

    def tent_predict(crop_bgr):
        net.load_state_dict(orig)  # reset mỗi crop
        for p in net.parameters(): p.requires_grad_(False)
        for p in gn_params: p.requires_grad_(True)
        opt = torch.optim.Adam(gn_params, lr=args.lr)
        cr = cv2.resize(crop_bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(make_channels(np.ascontiguousarray(cr), mode))[None].to(DEVICE)
        net.train()  # để norm cập nhật
        for _ in range(args.iters):
            lg = net(x); loss = ent(lg); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): pm = torch.sigmoid(net(x))[0, 0].float().cpu().numpy()
        return pm

    def union(bgr, boxes):
        H, W = bgr.shape[:2]; u = np.zeros((H, W), bool)
        for b in boxes:
            x0, y0, x1, y1 = pad_box(list(map(float, b)), 0.15, W, H)
            c = bgr[y0:y1, x0:x1]
            if c.size == 0: continue
            pm = tent_predict(c)
            pm = cv2.resize(pm, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > 0.5
            u[y0:y1, x0:x1] |= pm
        return u

    model = build_sam2("configs/sam2.1_hiera_t512", "checkpoints/sam2.1_hiera_tiny.pt",
                       device=DEVICE, hydra_overrides_extra=["++model.image_size=1024"])
    pred = SAM2ImagePredictor(model)
    dk = torch.load("checkpoints/detector_recall.pt", weights_only=False)
    det = DenseDetector(grid=dk.get("grid", 64)).to(DEVICE); det.load_state_dict(dk["det"]); det.eval()
    stems = json.load(open("labels_handdraw/select.json"))["stems"]
    have = [s for s in stems if os.path.isfile(f"labels_handdraw/masks/{s}.png")]
    rc = []; ra = []; grp = []
    for s in have:
        gt = cv2.imread(f"labels_handdraw/masks/{s}.png", 0) > 127
        bgr = cv2.imread(f"data/20241212/{s}.jpg"); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = gt.shape; grp.append(ncomp(gt)); spec, _, _ = clean_specimen(bgr)
        with torch.inference_mode(), AC:
            pred.set_image(rgb); feat = pred._features["image_embed"].float(); obj, bx = det(feat)
        ab = list(propose_boxes(obj[0].float(), bx[0].float(), H, W, spec=spec, thr=0.5))
        gb = frag_boxes(gt)
        rc.append(dice(union(bgr, gb) if gb else np.zeros_like(gt), gt))
        ra.append(dice(union(bgr, ab) if ab else np.zeros_like(gt), gt))
        print(f"  {s[:20]:22} ceil={rc[-1]:.3f} auto={ra[-1]:.3f}", flush=True)
    def md(x, mul=None):
        idx = range(len(x)) if mul is None else [i for i in range(len(x)) if (grp[i] > 1) == mul]
        return st.median([x[i] for i in idx]), st.mean([x[i] for i in idx])
    print(f"\n===== TENT (iters={args.iters}) trên 50 vẽ tay =====")
    print(f"CEILING   median={md(rc)[0]:.4f} mean={md(rc)[1]:.4f} | 1u={md(rc,False)[0]:.3f} >1u={md(rc,True)[0]:.3f}")
    print(f"FULL-AUTO median={md(ra)[0]:.4f} mean={md(ra)[1]:.4f} | 1u={md(ra,False)[0]:.3f} >1u={md(ra,True)[0]:.3f}")
    print(f"SO: segmenter (no-tent) ceiling 0.883 / full-auto 0.635")
    json.dump({"iters": args.iters, "ceiling_median": md(rc)[0], "full_auto_median": md(ra)[0]},
              open("results/tent_eval.json", "w"), indent=1)

if __name__ == "__main__":
    main()
