"""So Dice của SAM2(auto-box 0.7) vs SAM2(human-box GT) trên test set đóng băng.
Trả lời: nút Auto-box->SAM có dùng được không, hay chỉ nên làm prefill."""
import json, os, numpy as np, cv2, torch
ROOT=os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from propose_box import load_proposer, propose

DEV="cuda" if torch.cuda.is_available() else "cpu"
RES=int(os.environ.get("SAM2_RES","1024"))
m=build_sam2("configs/sam2.1_hiera_t512","checkpoints/sam2.1_hiera_tiny.pt",
             device=DEV, hydra_overrides_extra=[f"++model.image_size={RES}"])
PR=SAM2ImagePredictor(m)
AC=torch.autocast("cuda",dtype=torch.bfloat16) if DEV=="cuda" else torch.autocast("cpu",enabled=False)
P=load_proposer(); print("vote_thr",P["vote_thr"])

def dice(a,b):
    a=a.astype(bool); b=b.astype(bool); s=a.sum()+b.sum()
    return 1.0 if s==0 else 2*(a&b).sum()/s

def sam_box(rgb,box):
    with torch.inference_mode(), AC:
        PR.set_image(rgb)
        mk,sc,_=PR.predict(box=np.array(box,np.float32),multimask_output=True)
    bi=int(np.argmax(sc)); return mk[bi].astype(bool)

def gtbox(tm):
    ys,xs=np.where(tm); return [int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]

test=json.load(open("labels/test_frozen.json"))["test"]
rows=[]
for s in test:
    ip=f"data/20241212/{s}.jpg"; tp=f"labels/masks/{s}.png"
    if not (os.path.isfile(ip) and os.path.isfile(tp)): continue
    bgr=cv2.imread(ip); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    gt=(cv2.imread(tp,0)>127)
    if gt.sum()==0: continue
    ab=propose(P,bgr)
    hb=gtbox(gt.astype(np.uint8))
    d_h=dice(sam_box(rgb,hb),gt)
    if ab is None:
        rows.append((s,d_h,None,None,None)); continue
    m_a=sam_box(rgb,ab)
    d_a=dice(m_a,gt)
    spec_frac=m_a.sum()/gt.sum()   # mask auto / u thật: >>1 = nuốt mô lành
    rows.append((s,d_h,d_a,spec_frac,(ab[2]-ab[0])*(ab[3]-ab[1])))
print(f"\n{'stem':36} {'Dice(human)':>11} {'Dice(auto)':>11} {'mask/u':>8}")
dh=[];da=[];explode=0
for s,d_h,d_a,sf,_ in rows:
    da_s=f"{d_a:.3f}" if d_a is not None else "NOBOX"
    sf_s=f"{sf:.2f}x" if sf is not None else "-"
    print(f"{s[:36]:36} {d_h:>11.3f} {da_s:>11} {sf_s:>8}")
    dh.append(d_h)
    if d_a is not None:
        da.append(d_a)
        if sf>2.0: explode+=1
print(f"\nN={len(rows)} | Dice human-box median={np.median(dh):.3f} | "
      f"Dice auto-box median={np.median(da):.3f} mean={np.mean(da):.3f}")
print(f"Ca 'nổ' (mask auto > 2x u thật): {explode}/{len(da)}")
