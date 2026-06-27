"""
#I VLM-CRITIC (proof-of-concept) — Claude (VLM) tự nhìn overlay lưới toạ độ, đề xuất box u,
rồi segmenter (SegResNet+LAB) tô mask. So paired với detector full-auto trên ĐÚNG 12 ca.
Box do VLM đọc từ ảnh (toạ độ gốc). Env: medsam2_anno.
"""
import json, os, sys, numpy as np, cv2, torch, statistics as st
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import predict_seg_crop as P
from seg_crop import ncomp
from detector import DenseDetector, propose_boxes
from specimen_clean import clean_specimen
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
AC = torch.autocast("cuda", dtype=torch.bfloat16); DEVICE = "cuda"

# Box do VLM (Claude) đề xuất, toạ độ ẢNH GỐC [x0,y0,x1,y1]; khớp prefix stem
VLM = {
 "SS21-42826": [[1130, 660, 1620, 1040]],
 "SS22-49889": [[820, 520, 1015, 725], [1615, 695, 1845, 925]],
 "SS23-12955": [[1280, 150, 1645, 475]],
 "SS22-23641": [[950, 400, 1285, 905], [1545, 400, 1855, 955]],
 "SS21-34765": [[675, 605, 865, 775], [2035, 785, 2215, 965]],
 "SS22-02205": [[995, 995, 1205, 1210], [1735, 595, 1935, 835]],
 "SS23-61141": [[700, 360, 1065, 715], [1475, 225, 1865, 575]],
 "SS22-45604": [[870, 100, 1385, 905], [1600, 80, 2085, 855]],
 "SS21-58881": [[1240, 400, 1855, 1055]],
 "SS21-53209": [[150, 250, 715, 915], [1700, 250, 2360, 965]],
 "SS21-42835": [[1430, 830, 1665, 1065]],
 "SS23-45516": [[1255, 800, 1515, 990]],
}

def dice(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s

def main():
    stems = json.load(open("labels_handdraw/select.json"))["stems"]
    smap = {s[:18].split("^")[0]: s for s in stems}  # prefix patient -> full stem
    model = build_sam2("configs/sam2.1_hiera_t512", "checkpoints/sam2.1_hiera_tiny.pt",
                       device=DEVICE, hydra_overrides_extra=["++model.image_size=1024"])
    pred = SAM2ImagePredictor(model)
    dk = torch.load("checkpoints/detector_recall.pt", weights_only=False)
    det = DenseDetector(grid=dk.get("grid", 64)).to(DEVICE); det.load_state_dict(dk["det"]); det.eval()
    net, mode, _ = P.load_segmenter()
    rows = []
    for pidkey, boxes in VLM.items():
        full = smap.get(pidkey)
        if not full or not os.path.isfile(f"labels_handdraw/masks/{full}.png"):
            print(f"  bỏ {pidkey} (không có mask)"); continue
        gt = cv2.imread(f"labels_handdraw/masks/{full}.png", 0) > 127
        bgr = cv2.imread(f"data/20241212/{full}.jpg"); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = gt.shape; spec, _, _ = clean_specimen(bgr); no = ncomp(gt)
        # VLM box -> segmenter
        mv = P.boxes_to_mask(net, mode, DEVICE, bgr, boxes)
        # detector full-auto -> segmenter (paired)
        with torch.inference_mode(), AC:
            pred.set_image(rgb); feat = pred._features["image_embed"].float(); obj, bx = det(feat)
        ab = list(propose_boxes(obj[0].float(), bx[0].float(), H, W, spec=spec, thr=0.5))
        md = P.boxes_to_mask(net, mode, DEVICE, bgr, ab) if ab else np.zeros_like(gt)
        dv = dice(mv, gt); dd = dice(md, gt)
        rows.append((pidkey, no, dv, dd))
        print(f"  {pidkey:12} n={no} VLM={dv:.3f} detector={dd:.3f}  ({'VLM+' if dv>dd else 'det+'}{abs(dv-dd):.3f})", flush=True)
    nv = [r[2] for r in rows]; nd = [r[3] for r in rows]
    win = sum(1 for r in rows if r[2] > r[3] + 1e-3)
    print(f"\n===== VLM-CRITIC vs DETECTOR (n={len(rows)} ca) =====")
    print(f"VLM-box     median={st.median(nv):.4f} mean={st.mean(nv):.4f}")
    print(f"detector    median={st.median(nd):.4f} mean={st.mean(nd):.4f}")
    print(f"VLM thắng {win}/{len(rows)} ca")
    # tách nhóm u tách-bạch (theo nhận định VLM) vs mơ hồ
    json.dump({"vlm_median": st.median(nv), "vlm_mean": st.mean(nv),
               "det_median": st.median(nd), "det_mean": st.mean(nd),
               "rows": rows}, open("results/vlm_critic.json", "w"), indent=1)
    print("-> results/vlm_critic.json", flush=True)

if __name__ == "__main__":
    main()
