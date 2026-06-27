"""So mask deliverable (pure box->mask) vs mask curated, N=550. Dice + HD95."""
import os, glob, numpy as np, cv2, csv
ROOT=os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT)
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
rows=[]
for mp in sorted(glob.glob('eval_masks_550/*.png')):
    s=os.path.splitext(os.path.basename(mp))[0]
    gp=f'labels/masks/{s}.png'
    if not os.path.isfile(gp): continue
    pr=cv2.imread(mp,0)>127; gt=cv2.imread(gp,0)>127
    if gt.sum()==0: continue
    rows.append((s,dice(pr,gt),hd95(pr,gt),pr.sum()/max(1,gt.sum())))
D=np.array([r[1] for r in rows]); H=np.array([r[2] for r in rows]); R=np.array([r[3] for r in rows])
with open('results/deliverable_eval550.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['stem','dice','hd95_px','pred_over_gt'])
    for r in rows: w.writerow([r[0],round(r[1],4),round(r[2],2),round(r[3],3)])
print(f'N={len(rows)} (pure box->mask vs curated)')
print(f'Dice : median={np.median(D):.3f} mean={np.mean(D):.3f} | <0.8: {(D<0.8).sum()} | <0.5: {(D<0.5).sum()}')
print(f'HD95 : median={np.nanmedian(H):.1f}px mean={np.nanmean(H):.1f}px')
print(f'mask nổ >1.5x curated: {(R>1.5).sum()} ({100*(R>1.5).mean():.1f}%)')
for q in (0.1,0.25,0.5,0.75,0.9):
    print(f'  Dice phân vị {int(q*100)}%: {np.quantile(D,q):.3f}')
print('-> results/deliverable_eval550.csv')
