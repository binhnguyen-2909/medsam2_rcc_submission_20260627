"""
LOCALIZE #B — ANOMALY (novelty) detection. Train Autoencoder CHỈ trên patch MÔ LÀNH (không
dính u). Khi quét ảnh: vùng mô lành recon tốt (lỗi thấp), vùng U lạ -> recon lỗi cao -> heatmap
-> box. ⚠️ Vướng tường "u ~ mô lành cùng màu" nên lỗi có thể không bùng nổ; thử kiểm chứng.
  python anomaly_ae.py --epochs 40   (train AE)
anomaly_boxes(bgr,spec) dùng trong localize_eval. Env: medsam2_anno.
"""
import argparse, csv, json, os, sys, random, hashlib
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from seg_crop import set_seed, TRUTH_DIR, pat
from specimen_clean import clean_specimen
DEVICE = "cuda"; IMG_DIR = "data/20241212"; CKPT = "checkpoints/anomaly_ae.pt"; PSZ = 64

class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        def e(i,o): return nn.Sequential(nn.Conv2d(i,o,4,2,1), nn.GroupNorm(8,o), nn.GELU())
        def d(i,o): return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1), nn.GroupNorm(8,o), nn.GELU())
        self.enc = nn.Sequential(e(3,32), e(32,64), e(64,128), e(128,128))   # 64->4
        self.dec = nn.Sequential(d(128,128), d(128,64), d(64,32), nn.ConvTranspose2d(32,3,4,2,1))
    def forward(self,x): return self.dec(self.enc(x))

def patch_t(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    return torch.from_numpy(np.ascontiguousarray(((rgb-0.5)/0.5).transpose(2,0,1)))

def load_ae():
    net = ConvAE().to(DEVICE); net.load_state_dict(torch.load(CKPT, weights_only=False)["net"]); net.eval()
    return net

@torch.no_grad()
def anomaly_boxes(net, bgr, spec, thr_pct=92):
    H, W = bgr.shape[:2]; step = PSZ
    err = np.zeros((H, W), np.float32); wt = np.zeros((H, W), np.float32)
    batch=[]; locs=[]
    for y in range(0, H-PSZ+1, step):
        for x in range(0, W-PSZ+1, step):
            if spec[y+PSZ//2, x+PSZ//2]==0: continue
            batch.append(patch_t(bgr[y:y+PSZ, x:x+PSZ])); locs.append((x,y))
    for i in range(0, len(batch), 512):
        xb = torch.stack(batch[i:i+512]).to(DEVICE); rec = net(xb)
        e = ((rec-xb)**2).mean(dim=(1,2,3)).cpu().numpy()
        for j,(x,y) in enumerate(locs[i:i+512]):
            err[y:y+PSZ, x:x+PSZ]+=e[j]; wt[y:y+PSZ, x:x+PSZ]+=1
    err = err/np.maximum(wt,1e-6)
    vals = err[spec>0]
    if len(vals)==0: return []
    thr = np.percentile(vals, thr_pct)
    binm = ((err>thr)&(spec>0)).astype(np.uint8)
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, np.ones((25,25),np.uint8))
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((9,9),np.uint8))
    n,lab,st,_ = cv2.connectedComponentsWithStats(binm,8); boxes=[]
    thra = 0.003*binm.size
    for i in range(1,n):
        if st[i,cv2.CC_STAT_AREA]<thra: continue
        x0,y0=st[i,cv2.CC_STAT_LEFT],st[i,cv2.CC_STAT_TOP]
        boxes.append([float(x0),float(y0),float(x0+st[i,cv2.CC_STAT_WIDTH]),float(y0+st[i,cv2.CC_STAT_HEIGHT])])
    return boxes

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args(); set_seed(0)
    truth = sorted(f[:-4] for f in os.listdir(TRUTH_DIR) if f.endswith(".png"))
    hd = set(map(pat, json.load(open("labels_handdraw/select.json"))["stems"]))
    e200 = set(pat(r["stem"]) for r in csv.DictReader(open("results/confirm200_ft_vs_zs.csv")))
    t12 = set(map(pat, json.load(open("labels/test_frozen.json"))["test"]))
    clean = [s for s in truth if pat(s) not in (hd | e200 | t12)]
    print(f"[ae] trích patch MÔ LÀNH từ {len(clean)} ảnh...", flush=True)
    P=[]
    for s in clean:
        bgr = cv2.imread(f"{IMG_DIR}/{s}.jpg"); m = cv2.imread(f"{TRUTH_DIR}/{s}.png",0)>127
        spec,_,_ = clean_specimen(bgr); H,W = m.shape
        ys,xs = np.where(spec>0)
        if len(ys)==0: continue
        for _ in range(30):
            i=random.randint(0,len(ys)-1); cy,cx=ys[i],xs[i]
            y=max(0,min(H-PSZ,cy-PSZ//2)); x=max(0,min(W-PSZ,cx-PSZ//2))
            if m[y:y+PSZ, x:x+PSZ].mean()>0.01: continue   # bỏ patch dính u
            P.append(patch_t(bgr[y:y+PSZ, x:x+PSZ]))
    print(f"[ae] patch mô lành={len(P)}", flush=True)
    net = ConvAE().to(DEVICE); opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-5)
    for ep in range(1, args.epochs+1):
        net.train(); idx=torch.randperm(len(P)); tot=0; nb=0
        for i in range(0,len(idx),args.batch):
            b=idx[i:i+args.batch]; xb=torch.stack([P[j] for j in b]).to(DEVICE)
            rec=net(xb); loss=F.mse_loss(rec,xb)
            opt.zero_grad(); loss.backward(); opt.step(); tot+=float(loss); nb+=1
        if ep%10==0: print(f"[ep {ep}] recon_mse={tot/max(nb,1):.4f}", flush=True)
    torch.save({"net": net.state_dict()}, CKPT); print(f"[ae] -> {CKPT}", flush=True)

if __name__ == "__main__":
    main()
