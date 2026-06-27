"""
Trần của 'train cái đỏ (SAM)': với BOX detector THẬT hiện tại, nếu segmenter HOÀN HẢO
(mask = đúng phần u nằm trong union box, không tràn) thì full-auto Dice tối đa = bao nhiêu?
  C1 = Dice(GT ∩ union(box_detector), GT)  -> trần nếu chỉ cải thiện ĐỎ, giữ nguyên box.
So với: current auto (SAM zero-shot từ box) đã biết ~0.666.
Tách 1u/>1u. Cũng đo recall-trong-box vs tràn-ngoài-box để biết lỗi nào do đỏ.
"""
import os, sys, json, csv, numpy as np, cv2, torch
ROOT=os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0,ROOT)
from detector import DenseDetector
from specimen_clean import clean_specimen
from eval_handdraw import auto_boxes, sam_from_boxes, dice, ncomp
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
CFG="configs/sam2.1_hiera_t512"; CKPT="checkpoints/sam2.1_hiera_tiny.pt"
DET=os.environ.get("DET_CKPT","checkpoints/detector_recall.pt"); RES=1024
model=build_sam2(CFG,CKPT,device="cuda",hydra_overrides_extra=[f"++model.image_size={RES}"])
pred=SAM2ImagePredictor(model)
ck=torch.load(DET,weights_only=False); det=DenseDetector(grid=ck.get("grid",64)).to("cuda")
det.load_state_dict(ck["det"]); det.eval()
stems=json.load(open("labels_handdraw/select.json"))["stems"]
have=[s for s in stems if os.path.isfile(f"labels_handdraw/masks/{s}.png")]
rows=[]
for s in have:
    gt=cv2.imread(f"labels_handdraw/masks/{s}.png",0)>127
    bgr=cv2.imread(f"data/20241212/{s}.jpg"); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    spec,_,_=clean_specimen(bgr); H,W=gt.shape
    ab=auto_boxes(det,pred,rgb,0.5,spec=spec)
    boxreg=np.zeros((H,W),bool)
    for b in ab:
        x0,y0,x1,y1=[int(v) for v in b]; boxreg[max(0,y0):y1,max(0,x0):x1]=True
    perfect=gt&boxreg                      # segmenter hoàn hảo trong box
    c1=dice(perfect,gt)                    # trần nếu chỉ sửa ĐỎ
    red=sam_from_boxes(pred,rgb,ab) if len(ab) else np.zeros_like(gt)
    cur=dice(red,gt)
    gt_in=(gt&boxreg).sum(); gt_tot=gt.sum()
    recall_box=gt_in/max(1,gt_tot)         # % u được box phủ (giới hạn cứng của đỏ)
    spill=(red&~gt).sum()/max(1,red.sum()) # % đỏ nằm ngoài u (đỏ tô lố -> sửa được bằng train đỏ)
    rows.append((s,ncomp(gt),len(ab),cur,c1,recall_box,spill))
    print(f"{s[:24]:26} n={ncomp(gt)} box={len(ab)} cur={cur:.3f} redCeil={c1:.3f} boxRecall={recall_box:.2f} spill={spill:.2f}",flush=True)
import statistics as st
def med(i,f=lambda r:True):
    v=[r[i] for r in rows if f(r)]; return st.median(v),st.mean(v),len(v)
one=lambda r:r[1]<=1; mul=lambda r:r[1]>1
print("\n===== TRẦN TRAIN-ĐỎ (N=%d) ====="%len(rows))
for lab,i in [("current auto (SAM ZS)",3),("RED-CEILING (đỏ hoàn hảo, box giữ nguyên)",4),("box-recall (giới hạn cứng)",5),("spill đỏ ngoài-u",6)]:
    a=med(i);o=med(i,one);m=med(i,mul)
    print(f"{lab:42s} median={a[0]:.3f} mean={a[1]:.3f} | 1u={o[0]:.3f} >1u={m[0]:.3f}")
# gain tối đa từ train đỏ
cur=np.array([r[3] for r in rows]); c1=np.array([r[4] for r in rows])
print(f"\nGAIN tối đa nếu đỏ hoàn hảo: median {np.median(cur):.3f} -> {np.median(c1):.3f}  (+{np.median(c1)-np.median(cur):.3f})")
print(f"Box-recall median {np.median([r[5] for r in rows]):.2f}  => 1 - đó = phần u KHÔNG box nào phủ (train đỏ KHÔNG cứu được)")
