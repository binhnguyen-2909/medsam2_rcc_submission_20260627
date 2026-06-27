"""
SAM trả 3 mask/box. Hiện lấy argmax(score) -> hay dính 'cả tạng' (spill).
Đo tiềm năng chọn-mask KHÔN HƠN (không train), trên 50 vẽ tay, box DETECTOR thật:
  cur   = union các mask argmax(score)                       [= 0.666]
  small = union mask NHỎ NHẤT mỗi box (chống spill, heuristic suy diễn được lúc test)
  oracle= per-box chọn mask Dice-vs-GT tốt nhất (TRẦN của chọn-mask, cần GT)
"""
import os, sys, json, numpy as np, cv2, torch, statistics as st
ROOT=os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0,ROOT)
from detector import DenseDetector
from specimen_clean import clean_specimen
from eval_handdraw import auto_boxes, dice, ncomp
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
CFG="configs/sam2.1_hiera_t512"; CKPT="checkpoints/sam2.1_hiera_tiny.pt"; RES=1024
AC=torch.autocast("cuda",dtype=torch.bfloat16)
model=build_sam2(CFG,CKPT,device="cuda",hydra_overrides_extra=[f"++model.image_size={RES}"])
pred=SAM2ImagePredictor(model)
dck=torch.load("checkpoints/detector_recall.pt",weights_only=False)
det=DenseDetector(grid=dck.get("grid",64)).to("cuda"); det.load_state_dict(dck["det"]); det.eval()
stems=json.load(open("labels_handdraw/select.json"))["stems"]
have=[s for s in stems if os.path.isfile(f"labels_handdraw/masks/{s}.png")]
rows=[]
for s in have:
    gt=cv2.imread(f"labels_handdraw/masks/{s}.png",0)>127
    bgr=cv2.imread(f"data/20241212/{s}.jpg"); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    spec,_,_=clean_specimen(bgr); no=ncomp(gt); H,W=gt.shape
    ab=auto_boxes(det,pred,rgb,0.5,spec=spec)
    u_cur=np.zeros((H,W),bool); u_small=np.zeros((H,W),bool); u_or=np.zeros((H,W),bool)
    if len(ab):
        with torch.inference_mode(),AC:
            pred.set_image(rgb)
            for b in ab:
                mk,sc,_=pred.predict(box=b,multimask_output=True)  # 3 mask
                mk=mk.astype(bool)
                u_cur|=mk[int(np.argmax(sc))]
                areas=[m.sum() for m in mk]; u_small|=mk[int(np.argmin(areas))]
                dl=[ (2*(m&gt).sum()/(m.sum()+gt.sum()+1e-9)) for m in mk]
                u_or|=mk[int(np.argmax(dl))]
    rows.append((no, dice(u_cur,gt), dice(u_small,gt), dice(u_or,gt)))
    print(f"  {s[:22]:24} n={no} cur={rows[-1][1]:.3f} small={rows[-1][2]:.3f} oracle={rows[-1][3]:.3f}",flush=True)
def med(i,f=lambda r:True):
    v=[r[i] for r in rows if f(r)]; return st.median(v),st.mean(v)
one=lambda r:r[0]<=1; mul=lambda r:r[0]>1
print("\n===== CHỌN-MASK (N=%d), box detector thật ====="%len(rows))
for lab,i in [("cur argmax(score) [=0.666]",1),("smallest-mask (heuristic, dùng được)",2),("ORACLE best-Dice (trần chọn-mask)",3)]:
    a=med(i);o=med(i,one);m=med(i,mul)
    print(f"{lab:38s} median={a[0]:.4f} mean={a[1]:.4f} | 1u={o[0]:.3f} >1u={m[0]:.3f}")
