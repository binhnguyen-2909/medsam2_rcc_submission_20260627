"""Montage 200 ca, mỗi ca 3 ô: [GỐC | FT | ZS]. Overlay mask (fill) + viền GT (vàng).
Sắp theo delta (FT-ZS) tăng dần: FT-thua-nặng lên đầu. Box=cc-bbox curated."""
import json, os, csv, numpy as np, cv2, torch, gc
ROOT=os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
RES=1024; MIN_CC=1500; PW=340           # bề rộng mỗi ô
AC=torch.autocast("cuda",dtype=torch.bfloat16)
rows=sorted(csv.DictReader(open("results/confirm200_ft_vs_zs.csv")),key=lambda r:float(r["delta"]))
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
items=[]
for r in rows:
    s=r["stem"]; gt=cv2.imread(f'labels/masks/{s}.png',0)>127
    items.append((s,gt,cc_boxes(gt),float(r["dice_zs"]),float(r["dice_ft"])))
print(f"N={len(items)}")
def run(ckpt):
    m=build_sam2("configs/sam2.1_hiera_t512","checkpoints/sam2.1_hiera_tiny.pt",device="cuda",
                 hydra_overrides_extra=[f"++model.image_size={RES}"])
    if ckpt: m.load_state_dict(torch.load(ckpt,map_location="cuda",weights_only=False)["model"],strict=False)
    P=SAM2ImagePredictor(m); out=[]
    for s,gt,boxes,_,_ in items:
        rgb=cv2.cvtColor(cv2.imread(f'data/20241212/{s}.jpg'),cv2.COLOR_BGR2RGB); out.append(seg(P,rgb,boxes))
    del m,P; gc.collect(); torch.cuda.empty_cache(); return out
print("ZS..."); ZS=run(None); print("FT..."); FT=run("checkpoints/sam2.1_rcc_ft.pt")

def overlay(rgb, mask, color, gt):
    o=rgb.copy()
    if mask is not None: o[mask]=(0.45*o[mask]+0.55*np.array(color)).astype(np.uint8)
    gc_,_=cv2.findContours(gt.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o,gc_,-1,(255,255,0),4)         # GT vàng
    return o
def panel(img,label,sub):
    H=int(PW*img.shape[0]/img.shape[1]); im=cv2.resize(img,(PW,H))
    bar=np.full((26,PW,3),25,np.uint8)
    cv2.putText(bar,label,(5,18),cv2.FONT_HERSHEY_SIMPLEX,0.5,sub,1)
    return np.vstack([bar,im])

cases=[]
for i,(s,gt,boxes,dz,df) in enumerate(items):
    rgb=cv2.cvtColor(cv2.imread(f'data/20241212/{s}.jpg'),cv2.COLOR_BGR2RGB)
    ys,xs=np.where(gt); m=160
    y0,y1=max(0,ys.min()-m),min(rgb.shape[0],ys.max()+m); x0,x1=max(0,xs.min()-m),min(rgb.shape[1],xs.max()+m)
    sl=(slice(y0,y1),slice(x0,x1))
    p_o=panel(rgb[sl],"GOC",(255,255,255))
    p_f=panel(overlay(rgb,FT[i],(255,40,40),gt)[sl],f"FT  d={df:.3f}",(120,120,255))
    p_z=panel(overlay(rgb,ZS[i],(0,220,0),gt)[sl],f"ZS  d={dz:.3f}",(120,255,120))
    h=max(p_o.shape[0],p_f.shape[0],p_z.shape[0])
    pad=lambda p:np.vstack([p,np.full((h-p.shape[0],PW,3),25,np.uint8)])
    g=8; sep=np.full((h,g,3),60,np.uint8)
    cases.append(np.hstack([pad(p_o),sep,pad(p_f),sep,pad(p_z)]))

COLS=3
gridrows=[]
for i in range(0,len(cases),COLS):
    rw=cases[i:i+COLS]; h=max(c.shape[0] for c in rw)
    rw=[np.vstack([c,np.full((h-c.shape[0],c.shape[1],3),15,np.uint8)]) for c in rw]
    while len(rw)<COLS: rw.append(np.full((h,rw[0].shape[1],3),15,np.uint8))
    gap=np.full((h,16,3),15,np.uint8)
    out=rw[0]
    for c in rw[1:]: out=np.hstack([out,gap,c])
    gridrows.append(out)
W=max(r.shape[1] for r in gridrows)
gridrows=[np.hstack([r,np.full((r.shape[0],W-r.shape[1],3),15,np.uint8)]) for r in gridrows]
grid=np.vstack(gridrows)
leg=np.full((46,W,3),0,np.uint8)
cv2.putText(leg,"Moi ca 3 o: GOC | FT(do) | ZS(xanh) ; vien VANG=GT. Sap theo delta tang dan (FT thua nang -> dau).",
            (12,30),cv2.FONT_HERSHEY_SIMPLEX,0.62,(255,255,255),2)
grid=np.vstack([leg,grid])
cv2.imwrite("results/montage_orig_ft_zs.png",cv2.cvtColor(grid,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,4])
print("saved results/montage_orig_ft_zs.png", grid.shape)
