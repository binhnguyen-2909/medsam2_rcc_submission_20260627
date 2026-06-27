"""
EVAL "cái đỏ" trên 50 ảnh VẼ TAY (sạch, held-out). So SAM zero-shot vs SAM fine-tuned (RED_CKPT):
  (A) full-auto : box DETECTOR (champion) -> SAM -> Dice vs vẽ tay   [hiện ZS=0.666]
  (B) ceiling   : box = bbox từng MẢNH của mask vẽ tay -> SAM -> union -> Dice [hiện ZS=0.857]
Tách 1u/>1u. RED_CKPT rỗng -> chỉ đo zero-shot.
  RED_CKPT=checkpoints/sam2.1_rcc_red_truth.pt python eval_red_handdraw.py
"""
import os, sys, json, numpy as np, cv2, torch
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from detector import DenseDetector
from specimen_clean import clean_specimen
from eval_handdraw import auto_boxes, sam_from_boxes, dice, ncomp
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import statistics as st

CFG = "configs/sam2.1_hiera_t512"; CKPT = "checkpoints/sam2.1_hiera_tiny.pt"; RES = 1024
DET = os.environ.get("DET_CKPT", "checkpoints/detector_recall.pt")
RED = os.environ.get("RED_CKPT", "")
MIN_FRAC = 0.002

def frag_boxes(gt):
    n, lab, stx, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
    bs = []
    for i in range(1, n):
        if stx[i, cv2.CC_STAT_AREA] < MIN_FRAC * gt.size: continue
        x, y, w, h, _ = stx[i]; bs.append(np.array([x, y, x + w, y + h], np.float32))
    return bs

def main():
    model = build_sam2(CFG, CKPT, device="cuda", hydra_overrides_extra=[f"++model.image_size={RES}"])
    tag = "ZERO-SHOT"
    if RED and os.path.isfile(RED):
        ck = torch.load(RED, weights_only=False); model.load_state_dict(ck["model"])
        tag = f"FT({os.path.basename(RED)} ep{ck.get('epoch','?')} jit{ck.get('jitter','?')})"
    predictor = SAM2ImagePredictor(model)
    dck = torch.load(DET, weights_only=False)
    det = DenseDetector(grid=dck.get("grid", 64)).to("cuda"); det.load_state_dict(dck["det"]); det.eval()
    print(f"SAM={tag} | detector={DET}", flush=True)

    stems = json.load(open("labels_handdraw/select.json"))["stems"]
    have = [s for s in stems if os.path.isfile(f"labels_handdraw/masks/{s}.png")]
    rows = []
    for s in have:
        gt = cv2.imread(f"labels_handdraw/masks/{s}.png", 0) > 127
        bgr = cv2.imread(f"data/20241212/{s}.jpg"); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        spec, _, _ = clean_specimen(bgr); no = ncomp(gt)
        ab = auto_boxes(det, predictor, rgb, 0.5, spec=spec)
        d_auto = dice(sam_from_boxes(predictor, rgb, ab), gt) if len(ab) else 0.0
        fb = frag_boxes(gt)
        d_ceil = dice(sam_from_boxes(predictor, rgb, fb), gt) if fb else np.nan
        rows.append((no, d_auto, d_ceil))
        print(f"  {s[:24]:26} n={no} auto={d_auto:.3f} ceil={d_ceil:.3f}", flush=True)
    def med(i, f=lambda r: True):
        v = [r[i] for r in rows if f(r) and not (isinstance(r[i], float) and np.isnan(r[i]))]
        return (st.median(v), st.mean(v), len(v))
    one = lambda r: r[0] <= 1; mul = lambda r: r[0] > 1
    print(f"\n===== EVAL ĐỎ trên 50 vẽ tay — SAM={tag} =====")
    for lab, i in [("(A) full-auto (box detector)", 1), ("(B) ceiling (box GT-mảnh)", 2)]:
        a = med(i); o = med(i, one); m = med(i, mul)
        print(f"{lab:34s} median={a[0]:.4f} mean={a[1]:.4f} (n={a[2]}) | 1u={o[0]:.3f} >1u={m[0]:.3f}")
    print("BASELINE zero-shot: full-auto 0.666 | ceiling 0.857", flush=True)

if __name__ == "__main__":
    main()
