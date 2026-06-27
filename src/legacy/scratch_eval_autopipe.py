"""EVAL ĐÚNG: pipeline TỰ ĐỘNG end-to-end.
 - Auto-box proposer huấn luyện LẠI, LOẠI 200 ca eval (theo bệnh nhân) -> box HELD-OUT.
 - Mô hình tự đề xuất box -> SAM(ZS) & SAM(FT) sinh mask -> so Dice vs mask tay (labels/masks).
So với cách cũ (đưa bbox-chặt sẵn): đây đo CẢ localize + segment, FT vs ZS công bằng."""
import json, os, csv, numpy as np, cv2, torch, gc
ROOT=os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT)
from train_cellbox import build_dataset, train_ensemble, ensemble_vote, auto_box
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
VOTE_THR=0.6; RES=1024
AC=torch.autocast("cuda",dtype=torch.bfloat16)
def patient(s): return s.split("^")[0]
def dice(a,b):
    a=a.astype(bool);b=b.astype(bool);s=a.sum()+b.sum()
    return 1.0 if s==0 else 2*(a&b).sum()/s
def bbox_iou(a,b):
    if a is None or b is None: return 0.0
    ix0,iy0=max(a[0],b[0]),max(a[1],b[1]); ix1,iy1=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0,ix1-ix0),max(0,iy1-iy0); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

eval_stems=[r["stem"] for r in csv.DictReader(open("results/confirm200_ft_vs_zs.csv"))]
eval_pat=set(patient(s) for s in eval_stems)
done=json.load(open("labels/done.json"))
train_stems=[s for s in done if patient(s) not in eval_pat and os.path.isfile(f"labels/masks/{s}.png")]
print(f"eval={len(eval_stems)} ca / {len(eval_pat)} bệnh nhân | proposer train trên {len(train_stems)} ca (đã loại eval)")

# 1) huấn luyện proposer HELD-OUT
print("build_dataset train (proposer)..."); Xtr,Ytr,_,_,_=build_dataset(train_stems)
mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6
print(f"train {len(Xtr)} ô -> train 100 model..."); models=train_ensemble(Xtr,Ytr,mu,sd,n=100)

# 2) đề xuất box held-out cho 200 ca
print("build_dataset eval (features)..."); _,_,_,_,per=build_dataset(eval_stems)
auto={}
for s in eval_stems:
    rec=per.get(s)
    if rec is None: auto[s]=(None,None); continue
    valid=[ci for ci,v in enumerate(rec["valid"]) if v]
    if not valid: auto[s]=(None,rec.get("tumor_bbox")); continue
    X=np.array([rec["feat"][ci] for ci in valid],np.float32)
    vote,_=ensemble_vote(models,X,mu,sd)
    vbc={ci:float(v) for ci,v in zip(valid,vote)}
    box=auto_box(rec,vbc,vote_thr=VOTE_THR)
    auto[s]=(box,rec.get("tumor_bbox"))

# 3) SAM ZS & FT với auto-box
def seg(P,rgb,box):
    if box is None: return np.zeros(rgb.shape[:2],bool)
    with torch.inference_mode(),AC:
        P.set_image(rgb); mk,sc,_=P.predict(box=np.array(box,np.float32),multimask_output=True)
    return mk[int(np.argmax(sc))].astype(bool)
def run(ckpt):
    m=build_sam2("configs/sam2.1_hiera_t512","checkpoints/sam2.1_hiera_tiny.pt",device="cuda",
                 hydra_overrides_extra=[f"++model.image_size={RES}"])
    if ckpt: m.load_state_dict(torch.load(ckpt,map_location="cuda",weights_only=False)["model"],strict=False)
    P=SAM2ImagePredictor(m); out={}
    for s in eval_stems:
        rgb=cv2.cvtColor(cv2.imread(f"data/20241212/{s}.jpg"),cv2.COLOR_BGR2RGB)
        out[s]=dice(seg(P,rgb,auto[s][0]),cv2.imread(f"labels/masks/{s}.png",0)>127)
    del m,P; gc.collect(); torch.cuda.empty_cache(); return out
print("SAM zero-shot..."); DZ=run(None)
print("SAM fine-tuned..."); DF=run("checkpoints/sam2.1_rcc_ft.pt")

# multi-u + localize
import json as _j
def nobj(s):
    p=f"labels/prompts/{s}.json"; return _j.load(open(p)).get("n_objects",1) if os.path.isfile(p) else 1
rows=[]
for s in eval_stems:
    box,tb=auto[s]; iou=bbox_iou(box,tb)
    rows.append((s,DZ[s],DF[s],iou,1 if box else 0,nobj(s)))
with open("results/autopipe_ft_vs_zs.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["stem","dice_zs_auto","dice_ft_auto","box_iou_gt","box_found","n_obj"])
    for r in rows: w.writerow([r[0],round(r[1],4),round(r[2],4),round(r[3],4),r[4],r[5]])
Z=np.array([r[1] for r in rows]); F=np.array([r[2] for r in rows])
IOU=np.array([r[3] for r in rows]); NO=np.array([r[5] for r in rows]); d=F-Z
print(f"\n===== PIPELINE TỰ ĐỘNG (auto-box held-out -> SAM) vs mask tay, N={len(rows)} =====")
print(f"Box localize: IoU(auto-box, GT-bbox) median={np.median(IOU):.3f} | box trượt hẳn (IoU<0.1): {(IOU<0.1).sum()}")
print(f"Zero-shot  Dice: median={np.median(Z):.4f} mean={np.mean(Z):.4f}")
print(f"Fine-tuned Dice: median={np.median(F):.4f} mean={np.mean(F):.4f}")
print(f"delta FT-ZS: mean={d.mean():+.4f} | FT thắng/thua={int((d>1e-4).sum())}/{int((d<-1e-4).sum())}")
print(f"  ảnh 1 u (n={int((NO<=1).sum())}): ZS={np.median(Z[NO<=1]):.3f} FT={np.median(F[NO<=1]):.3f}")
print(f"  ảnh >1 u (n={int((NO>1).sum())}): ZS={np.median(Z[NO>1]):.3f} FT={np.median(F[NO>1]):.3f}")
print("-> results/autopipe_ft_vs_zs.csv")
