"""FT vs zero-shot, N=200, MỘT model/lượt (tránh OOM). Box=bbox tách-mảnh curated."""
import json, os, numpy as np, cv2, torch, csv, gc
ROOT=os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
N=200; SEED=777; MIN_CC=1500; RES=1024
AC=torch.autocast("cuda",dtype=torch.bfloat16)
def dice(a,b):
    a=a.astype(bool);b=b.astype(bool);s=a.sum()+b.sum()
    return 1.0 if s==0 else 2*(a&b).sum()/s
def cc_boxes(mask):
    n,_,st,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    return [[st[i,0],st[i,1],st[i,0]+st[i,2],st[i,1]+st[i,3]] for i in range(1,n) if st[i,4]>=MIN_CC]
def seg(P,rgb,boxes):
    H,W=rgb.shape[:2]; u=np.zeros((H,W),bool)
    with torch.inference_mode(), AC:
        P.set_image(rgb)
        for b in boxes:
            mk,sc,_=P.predict(box=np.array(b,np.float32),multimask_output=True)
            u|=mk[int(np.argmax(sc))].astype(bool)
    return u
done=sorted(json.load(open('labels/done.json')))
test=set(json.load(open('labels/test_frozen.json'))['test'])
pool=[s for s in done if s not in test and os.path.isfile(f'labels/masks/{s}.png')]
rng=np.random.default_rng(SEED); sample=sorted(rng.choice(pool,min(N,len(pool)),replace=False))
# nạp trước GT + box + ảnh-path (CPU)
items=[]
for s in sample:
    gt=cv2.imread(f'labels/masks/{s}.png',0)>127
    if gt.sum()==0: continue
    b=cc_boxes(gt)
    if b: items.append((s,gt,b))
print(f"N={len(items)} (seed {SEED})")
def run(ckpt_path):
    m=build_sam2("configs/sam2.1_hiera_t512","checkpoints/sam2.1_hiera_tiny.pt",
                 device="cuda", hydra_overrides_extra=[f"++model.image_size={RES}"])
    if ckpt_path:
        ck=torch.load(ckpt_path,map_location="cuda",weights_only=False)
        m.load_state_dict(ck["model"],strict=False)
    P=SAM2ImagePredictor(m); out={}
    for s,gt,boxes in items:
        rgb=cv2.cvtColor(cv2.imread(f'data/20241212/{s}.jpg'),cv2.COLOR_BGR2RGB)
        out[s]=dice(seg(P,rgb,boxes),gt)
    del m,P; gc.collect(); torch.cuda.empty_cache()
    return out
print("lượt 1: zero-shot..."); ZS=run(None)
print("lượt 2: fine-tuned..."); FT=run("checkpoints/sam2.1_rcc_ft.pt")
Z=np.array([ZS[s] for s,_,_ in items]); F=np.array([FT[s] for s,_,_ in items]); d=F-Z
with open('results/confirm200_ft_vs_zs.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['stem','dice_zs','dice_ft','delta'])
    for s,_,_ in items: w.writerow([s,round(ZS[s],4),round(FT[s],4),round(FT[s]-ZS[s],4)])
print(f"\n=== N={len(items)} (FT epoch6 vs zero-shot, vs curated GT) ===")
print(f"Zero-shot : median={np.median(Z):.4f} mean={np.mean(Z):.4f}")
print(f"Fine-tuned: median={np.median(F):.4f} mean={np.mean(F):.4f}")
print(f"Delta FT-ZS: mean={d.mean():+.4f} median={np.median(d):+.4f}")
print(f"FT thắng/hòa/thua: {(d>1e-4).sum()}/{(np.abs(d)<=1e-4).sum()}/{(d<-1e-4).sum()}")
print(f"FT hơn rõ >+0.02: {(d>0.02).sum()} | tệ rõ <-0.02: {(d<-0.02).sum()}")
print('-> results/confirm200_ft_vs_zs.csv')
