"""Montage trực quan: GT(vàng)/ZeroShot(xanh)/FT(đỏ) trên ca FT-thua-nặng & FT-thắng."""
import json, os, csv, numpy as np, cv2, torch, gc
ROOT=os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
RES=1024; MIN_CC=1500
AC=torch.autocast("cuda",dtype=torch.bfloat16)
rows=sorted(csv.DictReader(open("results/confirm200_ft_vs_zs.csv")),key=lambda r:float(r["delta"]))
pick=rows[:4]+rows[-2:]          # 4 ca FT thua nặng nhất + 2 ca FT thắng nhất
def cc_boxes(m):
    n,_,st,_=cv2.connectedComponentsWithStats(m.astype(np.uint8),8)
    return [[st[i,0],st[i,1],st[i,0]+st[i,2],st[i,1]+st[i,3]] for i in range(1,n) if st[i,4]>=MIN_CC]
def seg(P,rgb,boxes):
    H,W=rgb.shape[:2]; u=np.zeros((H,W),bool)
    with torch.inference_mode(),AC:
        P.set_image(rgb)
        for b in boxes:
            mk,sc,_=P.predict(box=np.array(b,np.float32),multimask_output=True); u|=mk[int(np.argmax(sc))].astype(bool)
    return u
data=[]
for r in pick:
    s=r["stem"]; gt=cv2.imread(f'labels/masks/{s}.png',0)>127
    data.append((s,gt,cc_boxes(gt),float(r["dice_zs"]),float(r["dice_ft"])))
def run(ckpt):
    m=build_sam2("configs/sam2.1_hiera_t512","checkpoints/sam2.1_hiera_tiny.pt",device="cuda",
                 hydra_overrides_extra=[f"++model.image_size={RES}"])
    if ckpt: m.load_state_dict(torch.load(ckpt,map_location="cuda",weights_only=False)["model"],strict=False)
    P=SAM2ImagePredictor(m); out=[]
    for s,gt,boxes,_,_ in data:
        rgb=cv2.cvtColor(cv2.imread(f'data/20241212/{s}.jpg'),cv2.COLOR_BGR2RGB); out.append(seg(P,rgb,boxes))
    del m,P; gc.collect(); torch.cuda.empty_cache(); return out
ZS=run(None); FT=run("checkpoints/sam2.1_rcc_ft.pt")
tiles=[]
for i,(s,gt,boxes,dz,df) in enumerate(data):
    rgb=cv2.cvtColor(cv2.imread(f'data/20241212/{s}.jpg'),cv2.COLOR_BGR2RGB).copy()
    ys,xs=np.where(gt); m=120
    x0,x1=max(0,xs.min()-m),min(rgb.shape[1],xs.max()+m); y0,y1=max(0,ys.min()-m),min(rgb.shape[0],ys.max()+m)
    for mask,col in [(gt,(255,255,0)),(ZS[i],(0,220,0)),(FT[i],(255,40,40))]:
        c,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb,c,-1,col,5)
    crop=rgb[y0:y1,x0:x1]; crop=cv2.resize(crop,(420,int(420*crop.shape[0]/crop.shape[1])))
    bar=np.full((44,crop.shape[1],3),30,np.uint8)
    cv2.putText(bar,f"ZS={dz:.3f}  FT={df:.3f}  d={df-dz:+.3f}",(6,30),cv2.FONT_HERSHEY_SIMPLEX,0.62,
                ((0,255,0) if df>=dz else (80,80,255)),2)
    tiles.append(np.vstack([bar,crop]))
h=max(t.shape[0] for t in tiles); tiles=[np.vstack([t,np.full((h-t.shape[0],t.shape[1],3),30,np.uint8)]) for t in tiles]
cols=3; rowsN=[tiles[i:i+cols] for i in range(0,len(tiles),cols)]
gridrows=[np.hstack(rw+[np.full((rw[0].shape[0],420,3),30,np.uint8)]*(cols-len(rw))) for rw in rowsN]
grid=np.vstack(gridrows)
leg=np.full((50,grid.shape[1],3),20,np.uint8)
cv2.putText(leg,"GT=vang  ZeroShot=xanh  FT=do  | 4 ca dau: FT thua nang | 2 ca cuoi: FT thang",
            (10,33),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
grid=np.vstack([leg,grid])
cv2.imwrite("results/confirm200_qualitative.png",cv2.cvtColor(grid,cv2.COLOR_RGB2BGR))
print("saved results/confirm200_qualitative.png", grid.shape)
