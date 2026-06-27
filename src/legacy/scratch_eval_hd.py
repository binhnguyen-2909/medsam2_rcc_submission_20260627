"""Dice + HD + HD95 (biên) cho deliverable zero-shot box->mask trên 12 ảnh test.
Box = bbox của GT mask (giả lập người vẽ box quanh u). Đơn vị: pixel."""
import json, os, numpy as np, cv2, torch
ROOT=os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
DEV="cuda" if torch.cuda.is_available() else "cpu"; RES=1024
m=build_sam2("configs/sam2.1_hiera_t512","checkpoints/sam2.1_hiera_tiny.pt",
             device=DEV, hydra_overrides_extra=[f"++model.image_size={RES}"])
PR=SAM2ImagePredictor(m)
AC=torch.autocast("cuda",dtype=torch.bfloat16) if DEV=="cuda" else torch.autocast("cpu",enabled=False)

def dice(a,b):
    a=a.astype(bool);b=b.astype(bool);s=a.sum()+b.sum()
    return 1.0 if s==0 else 2*(a&b).sum()/s

def boundary(mask):
    mu=mask.astype(np.uint8)
    er=cv2.erode(mu,np.ones((3,3),np.uint8))
    return (mu-er)>0

def hd_metrics(pred,gt):
    """HD (max) và HD95 (px) hai chiều, qua distance transform."""
    pb=boundary(pred); gb=boundary(gt)
    if pb.sum()==0 or gb.sum()==0: return np.nan, np.nan
    # distanceTransform: khoảng cách tới pixel 0 gần nhất -> đặt biên = 0
    dt_g=cv2.distanceTransform((~gb).astype(np.uint8),cv2.DIST_L2,3)
    dt_p=cv2.distanceTransform((~pb).astype(np.uint8),cv2.DIST_L2,3)
    d_pg=dt_g[pb]   # pred-biên -> gt-biên
    d_gp=dt_p[gb]   # gt-biên  -> pred-biên
    alld=np.concatenate([d_pg,d_gp])
    return float(max(d_pg.max(),d_gp.max())), float(np.percentile(alld,95))

def gtbox(tm):
    ys,xs=np.where(tm); return [int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]

test=json.load(open("labels/test_frozen.json"))["test"]
rows=[]
for s in test:
    ip=f"data/20241212/{s}.jpg"; tp=f"labels/masks/{s}.png"
    if not (os.path.isfile(ip) and os.path.isfile(tp)): continue
    gt=cv2.imread(tp,0)>127
    if gt.sum()==0: continue
    rgb=cv2.cvtColor(cv2.imread(ip),cv2.COLOR_BGR2RGB)
    H,W=rgb.shape[:2]; diag=(H*H+W*W)**0.5
    with torch.inference_mode(), AC:
        PR.set_image(rgb)
        mk,sc,_=PR.predict(box=np.array(gtbox(gt.astype(np.uint8)),np.float32),multimask_output=True)
    pr=mk[int(np.argmax(sc))].astype(bool)
    d=dice(pr,gt); hd,hd95=hd_metrics(pr,gt)
    rows.append((s,d,hd,hd95,hd/diag*100,hd95/diag*100))
print(f"{'stem':32}{'Dice':>7}{'HD(px)':>9}{'HD95(px)':>10}{'HD%diag':>9}{'HD95%diag':>10}")
for s,d,hd,hd95,hdp,hd95p in rows:
    print(f"{s[:32]:32}{d:>7.3f}{hd:>9.1f}{hd95:>10.1f}{hdp:>8.2f}%{hd95p:>9.2f}%")
D=np.array([r[1] for r in rows]); HD=np.array([r[2] for r in rows]); H95=np.array([r[3] for r in rows])
print(f"\nN={len(rows)}")
print(f"Dice : median={np.median(D):.3f} mean={np.mean(D):.3f}")
print(f"HD   : median={np.median(HD):.1f}px mean={np.mean(HD):.1f}px  (max ảnh={HD.max():.0f}px)")
print(f"HD95 : median={np.median(H95):.1f}px mean={np.mean(H95):.1f}px")
print(f"(ảnh ~2736x1824, đường chéo ~3289px -> HD95 median = {np.median(H95)/3289*100:.2f}% đường chéo)")
