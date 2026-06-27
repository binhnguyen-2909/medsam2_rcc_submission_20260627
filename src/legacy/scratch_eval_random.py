"""Test box->mask trên MẪU NGẪU NHIÊN N ảnh (ngoài 12 frozen). Box input = bbox
tách-mảnh (connected-component) của mask curated -> xử lý được cả ảnh nhiều mảnh.
So Dice + HD95 vs curated. (Caveat: GT do SAM hỗ trợ tạo -> số thiên vị.)"""
import json, os, numpy as np, cv2, torch, csv
ROOT=os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
N=150; SEED=20260621; MIN_CC=1500   # bỏ mảnh < MIN_CC px (nhiễu)
DEV="cuda" if torch.cuda.is_available() else "cpu"; RES=1024
m=build_sam2("configs/sam2.1_hiera_t512","checkpoints/sam2.1_hiera_tiny.pt",
             device=DEV, hydra_overrides_extra=[f"++model.image_size={RES}"])
PR=SAM2ImagePredictor(m)
AC=torch.autocast("cuda",dtype=torch.bfloat16) if DEV=="cuda" else torch.autocast("cpu",enabled=False)
def dice(a,b):
    a=a.astype(bool);b=b.astype(bool);s=a.sum()+b.sum()
    return 1.0 if s==0 else 2*(a&b).sum()/s
def boundary(m):
    mu=m.astype(np.uint8); return (mu-cv2.erode(mu,np.ones((3,3),np.uint8)))>0
def hd95(pred,gt):
    pb=boundary(pred); gb=boundary(gt)
    if pb.sum()==0 or gb.sum()==0: return np.nan
    dg=cv2.distanceTransform((~gb).astype(np.uint8),cv2.DIST_L2,3)
    dp=cv2.distanceTransform((~pb).astype(np.uint8),cv2.DIST_L2,3)
    return float(np.percentile(np.concatenate([dg[pb],dp[gb]]),95))
def cc_boxes(mask):
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    out=[]
    for i in range(1,n):
        x,y,w,h,a=stats[i]
        if a>=MIN_CC: out.append([x,y,x+w,y+h])
    return out
done=sorted(json.load(open('labels/done.json')))
test=set(json.load(open('labels/test_frozen.json'))['test'])
pool=[s for s in done if s not in test and os.path.isfile(f'labels/masks/{s}.png')]
rng=np.random.default_rng(SEED); sample=sorted(rng.choice(pool,min(N,len(pool)),replace=False))
print(f"pool={len(pool)} -> mẫu ngẫu nhiên N={len(sample)} (seed {SEED})")
rows=[]
for s in sample:
    gt=cv2.imread(f'labels/masks/{s}.png',0)>127
    if gt.sum()==0: continue
    boxes=cc_boxes(gt)
    if not boxes: continue
    rgb=cv2.cvtColor(cv2.imread(f'data/20241212/{s}.jpg'),cv2.COLOR_BGR2RGB)
    H,W=rgb.shape[:2]; union=np.zeros((H,W),bool)
    with torch.inference_mode(), AC:
        PR.set_image(rgb)
        for b in boxes:
            mk,sc,_=PR.predict(box=np.array(b,np.float32),multimask_output=True)
            union|=mk[int(np.argmax(sc))].astype(bool)
    rows.append((s,len(boxes),dice(union,gt),hd95(union,gt)))
D=np.array([r[2] for r in rows]); Hd=np.array([r[3] for r in rows])
with open('results/eval_random150.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['stem','n_cc','dice','hd95_px'])
    for r in rows: w.writerow([r[0],r[1],round(r[2],4),round(r[3],2)])
print(f"\nN={len(rows)} (box=bbox tách-mảnh, vs curated)")
print(f"Dice : median={np.median(D):.3f} mean={np.mean(D):.3f}")
for q in (0.1,0.25,0.5,0.75,0.9): print(f"  Dice p{int(q*100)}: {np.quantile(D,q):.3f}")
print(f"Dice <0.9: {(D<0.9).sum()} | <0.8: {(D<0.8).sum()} | <0.5: {(D<0.5).sum()}")
print(f"HD95 : median={np.nanmedian(Hd):.1f}px mean={np.nanmean(Hd):.1f}px")
worst=sorted(rows,key=lambda r:r[2])[:6]
print("6 ca thấp nhất:"); [print(f"  {s[:32]:32} cc={n} Dice={d:.3f} HD95={h:.0f}px") for s,n,d,h in worst]
print('-> results/eval_random150.csv')
