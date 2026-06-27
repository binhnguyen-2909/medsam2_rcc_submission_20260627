"""So Dice SAM(auto-box) ở NHIỀU vote_thr với model 1020-nhãn vs human-box."""
import json, os, numpy as np, cv2, torch
ROOT=os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from propose_box import load_proposer, propose
DEV="cuda"; RES=1024
m=build_sam2("configs/sam2.1_hiera_t512","checkpoints/sam2.1_hiera_tiny.pt",
             device=DEV, hydra_overrides_extra=[f"++model.image_size={RES}"])
PR=SAM2ImagePredictor(m)
AC=torch.autocast("cuda",dtype=torch.bfloat16)
P=load_proposer()
def dice(a,b):
    a=a.astype(bool);b=b.astype(bool);s=a.sum()+b.sum()
    return 1.0 if s==0 else 2*(a&b).sum()/s
def sam_box(rgb,box):
    with torch.inference_mode(), AC:
        PR.set_image(rgb); mk,sc,_=PR.predict(box=np.array(box,np.float32),multimask_output=True)
    return mk[int(np.argmax(sc))].astype(bool)
def gtbox(tm):
    ys,xs=np.where(tm); return [int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]
test=json.load(open("labels/test_frozen.json"))["test"]
imgs=[]
for s in test:
    ip=f"data/20241212/{s}.jpg"; tp=f"labels/masks/{s}.png"
    if os.path.isfile(ip) and os.path.isfile(tp):
        gt=(cv2.imread(tp,0)>127)
        if gt.sum()>0: imgs.append((s,cv2.imread(ip),gt))
# human-box baseline (1 lần)
dh=[]
for s,bgr,gt in imgs:
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    dh.append(dice(sam_box(rgb,gtbox(gt.astype(np.uint8))),gt))
print(f"BASELINE human-box: Dice median={np.median(dh):.3f} (N={len(imgs)})")
print(f"\n{'vote':>5} {'Dice_auto_med':>13} {'mean':>6} {'nổ>2x':>6} {'Dice=0':>7}")
for vt in [0.5,0.6,0.7,0.8,0.9]:
    P["vote_thr"]=vt
    da=[];expl=0;z=0
    for s,bgr,gt in imgs:
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        ab=propose(P,bgr)
        if ab is None: continue
        ma=sam_box(rgb,ab); d=dice(ma,gt); da.append(d)
        if ma.sum()/gt.sum()>2.0: expl+=1
        if d<0.05: z+=1
    print(f"{vt:>5.1f} {np.median(da):>13.3f} {np.mean(da):>6.3f} {expl:>4}/{len(da)} {z:>5}/{len(da)}")
