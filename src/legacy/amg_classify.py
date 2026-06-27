"""
LOCALIZE #A — AMG + classifier. SAM rải lưới điểm -> hàng trăm mask phủ mọi thứ (mỡ/vỏ/tủy/u)
-> crop từng mask -> ResNet18 phân loại U/không-U -> giữ conf>thr -> box. Ranh giới mask theo
hình thức tự nhiên (có thể khít hơn box detector). Train classifier trên 178 mask thật.
  python amg_classify.py --epochs 25   (train classifier)
amg_boxes(bgr,spec) dùng trong localize_eval. Env: medsam2_anno.
"""
import argparse, csv, json, os, sys, random, hashlib
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from seg_crop import set_seed, TRUTH_DIR, pat
from specimen_clean import clean_specimen
DEVICE = "cuda"; IMG_DIR = "data/20241212"; CKPT = "checkpoints/amg_clf.pt"; CSIZE = 160
AC = torch.autocast("cuda", dtype=torch.bfloat16)
MINF = 0.0015; MAXF = 0.35  # mask area / specimen area: bỏ quá nhỏ / quá to (cả lát)

def build_clf():
    import torchvision.models as M
    net = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net

def crop_norm(bgr, box):
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(bgr.shape[1], x1); y1 = min(bgr.shape[0], y1)
    if x1 <= x0 or y1 <= y0: return None
    cr = cv2.resize(bgr[y0:y1, x0:x1], (CSIZE, CSIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(cr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([.485, .456, .406])) / np.array([.229, .224, .225])
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()

def load_clf():
    net = build_clf().to(DEVICE); net.load_state_dict(torch.load(CKPT, weights_only=False)["net"]); net.eval()
    return net

# ---- AMG bằng lưới điểm (SAM2 không có sẵn AutomaticMaskGenerator) ----
@torch.no_grad()
def amg_masks(predictor, rgb, spec, grid=16):
    H, W = rgb.shape[:2]; predictor.set_image(rgb)
    ys = np.linspace(0, H - 1, grid).astype(int); xs = np.linspace(0, W - 1, grid).astype(int)
    pts = [(x, y) for y in ys for x in xs if spec[y, x] > 0]
    masks = []
    spec_area = max(1, int((spec > 0).sum()))
    for (x, y) in pts:
        with AC:
            mk, sc, _ = predictor.predict(point_coords=np.array([[x, y]], np.float32),
                                          point_labels=np.array([1], np.int32), multimask_output=True)
        m = mk[int(np.argmax(sc))].astype(bool)
        a = m.sum()
        if a < MINF * spec_area or a > MAXF * spec_area: continue
        masks.append(m)
    # NMS theo IoU để gộp trùng
    kept = []
    for m in sorted(masks, key=lambda z: -z.sum()):
        dup = False
        for k in kept:
            inter = (m & k).sum(); uni = (m | k).sum()
            if uni > 0 and inter / uni > 0.7: dup = True; break
        if not dup: kept.append(m)
    return kept

@torch.no_grad()
def amg_boxes(predictor, net, rgb, bgr, spec, conf=0.9, grid=16):
    masks = amg_masks(predictor, rgb, spec, grid)
    boxes = []
    for m in masks:
        ys, xs = np.where(m)
        if len(ys) == 0: continue
        box = [xs.min(), ys.min(), xs.max(), ys.max()]
        t = crop_norm(bgr, box)
        if t is None: continue
        p = torch.sigmoid(net(t[None].to(DEVICE)))[0, 0].item()
        if p > conf: boxes.append([float(b) for b in box])
    return boxes

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args(); set_seed(0)
    truth = sorted(f[:-4] for f in os.listdir(TRUTH_DIR) if f.endswith(".png"))
    hd = set(map(pat, json.load(open("labels_handdraw/select.json"))["stems"]))
    e200 = set(pat(r["stem"]) for r in csv.DictReader(open("results/confirm200_ft_vs_zs.csv")))
    t12 = set(map(pat, json.load(open("labels/test_frozen.json"))["test"]))
    clean = [s for s in truth if pat(s) not in (hd | e200 | t12)]
    print(f"[amg-clf] dựng crops từ {len(clean)} ảnh...", flush=True)
    X, Y = [], []
    from seg_crop import frag_boxes
    for s in clean:
        bgr = cv2.imread(f"{IMG_DIR}/{s}.jpg"); m = cv2.imread(f"{TRUTH_DIR}/{s}.png", 0) > 127
        spec, _, _ = clean_specimen(bgr); H, W = m.shape
        # dương: mảnh u (+jitter nhẹ)
        for b in frag_boxes(m):
            for _ in range(3):
                bw = b[2]-b[0]; bh = b[3]-b[1]; j = 0.12
                bb = [b[0]+random.uniform(-j,j)*bw, b[1]+random.uniform(-j,j)*bh,
                      b[2]+random.uniform(-j,j)*bw, b[3]+random.uniform(-j,j)*bh]
                t = crop_norm(bgr, bb)
                if t is not None: X.append(t); Y.append(1.0)
        # âm: ô ngẫu nhiên trong specimen KHÔNG dính u
        ys, xs = np.where(spec > 0)
        if len(ys) == 0: continue
        for _ in range(8):
            i = random.randint(0, len(ys)-1); cy, cx = ys[i], xs[i]
            sz = random.randint(120, 500)
            bb = [cx-sz, cy-sz, cx+sz, cy+sz]
            x0,y0,x1,y1 = [max(0,int(bb[0])),max(0,int(bb[1])),min(W,int(bb[2])),min(H,int(bb[3]))]
            if m[y0:y1, x0:x1].mean() > 0.03: continue
            t = crop_norm(bgr, [x0,y0,x1,y1])
            if t is not None: X.append(t); Y.append(0.0)
    Y = np.array(Y, np.float32)
    print(f"[amg-clf] crops={len(X)} pos={int(Y.sum())} ({100*Y.mean():.1f}%)", flush=True)
    net = build_clf().to(DEVICE); opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    posw = torch.tensor([(Y==0).sum()/max(1,(Y==1).sum())]).to(DEVICE)
    Yt = torch.from_numpy(Y).to(DEVICE)[:, None]
    for ep in range(1, args.epochs+1):
        net.train(); idx = torch.randperm(len(X)); tot=0; nb=0
        for i in range(0, len(idx), args.batch):
            b = idx[i:i+args.batch]; xb = torch.stack([X[j] for j in b]).to(DEVICE)
            lg = net(xb); loss = F.binary_cross_entropy_with_logits(lg, Yt[b], pos_weight=posw)
            opt.zero_grad(); loss.backward(); opt.step(); tot+=float(loss); nb+=1
        if ep%5==0: print(f"[ep {ep}] loss={tot/max(nb,1):.4f}", flush=True)
    torch.save({"net": net.state_dict()}, CKPT); print(f"[amg-clf] -> {CKPT}", flush=True)

if __name__ == "__main__":
    main()
