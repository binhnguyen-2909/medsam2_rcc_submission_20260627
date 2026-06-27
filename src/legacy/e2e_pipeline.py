"""PIPELINE TỰ ĐỘNG end-to-end, ĐÁNH GIÁ ĐÚNG (auto multi-box -> SAM -> vs mask tay).
KHÔNG RÒ RỈ: 200 ca eval tách khỏi CẢ SAM-finetune LẪN proposer.

Các bước (chạy sau khi đã có >=25GB GPU, do wrapper gate):
 1. Chia: eval200 (test cuối, không model nào thấy) | val_sel (chọn epoch FT) | train (còn lại)
 2. Finetune SAM held-out trên train, batch lớn (25GB) -> checkpoints/sam2.1_rcc_ft_e2e.pt
 3. Retrain proposer trên (mọi ca ngoài eval200) -> auto-box held-out
 4. Auto-box MULTI (mọi khối u) cho eval200
 5. SAM zero-shot & SAM FT(e2e) với auto-box -> GỘP mask -> Dice vs mask tay
 6. Báo cáo + montage [GỐC | FT | ZS]
"""
import json, os, csv, subprocess, hashlib, gc
import numpy as np, cv2, torch
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT)
from train_cellbox import build_dataset, train_ensemble, ensemble_vote, auto_box
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
PY = "/home/hvusynh2/conda_envs/medsam2_anno/bin/python"
VOTE_THR = 0.6; RES = 1024; MIN_CC = 1500
AC = torch.autocast("cuda", dtype=torch.bfloat16)
FT_CKPT = "checkpoints/sam2.1_rcc_ft_e2e.pt"
def patient(s): return s.split("^")[0]
def dice(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s
def hd95(pred, gt):
    def bd(m): mu = m.astype(np.uint8); return (mu - cv2.erode(mu, np.ones((3, 3), np.uint8))) > 0
    pb, gb = bd(pred), bd(gt)
    if pb.sum() == 0 or gb.sum() == 0: return np.nan
    dg = cv2.distanceTransform((~gb).astype(np.uint8), cv2.DIST_L2, 3)
    dp = cv2.distanceTransform((~pb).astype(np.uint8), cv2.DIST_L2, 3)
    return float(np.percentile(np.concatenate([dg[pb], dp[gb]]), 95))

# ---------- 1. chia tập (không rò rỉ) ----------
eval200 = [r["stem"] for r in csv.DictReader(open("results/confirm200_ft_vs_zs.csv"))]
epat = set(map(patient, eval200))
done = [s for s in json.load(open("labels/done.json")) if os.path.isfile(f"labels/masks/{s}.png")]
remaining = [s for s in done if patient(s) not in epat]          # proposer train (ngoài eval200)
rpat = sorted(set(map(patient, remaining)), key=lambda p: hashlib.md5(p.encode()).hexdigest())
val_pat = set(rpat[:10])
val_sel = sorted(s for s in remaining if patient(s) in val_pat)  # chọn epoch FT
train = sorted(s for s in remaining if patient(s) not in val_pat)
json.dump({"train": train, "val": val_sel}, open("labels/split_e2e.json", "w"))
print(f"[split] eval200={len(eval200)} | train(SAM)={len(train)} | val_sel={len(val_sel)} | "
      f"proposer_train(remaining)={len(remaining)} | 0 rò rỉ bệnh nhân eval")

# ---------- 2. finetune SAM held-out (batch lớn) ----------
print("[2] finetune SAM held-out (batch 128, precompute 32)...")
subprocess.run([PY, "-u", "finetune_sam2.py", "--split", "labels/split_e2e.json",
                "--ckpt_out", FT_CKPT, "--epochs", "40", "--eval_every", "2",
                "--batch", "32", "--precompute_batch", "8"], check=True)

# ---------- 3. retrain proposer held-out ----------
print("[3] build_dataset proposer-train (held-out)..."); Xtr, Ytr, _, _, _ = build_dataset(remaining)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
print(f"   {len(Xtr)} ô -> train 100 model"); models = train_ensemble(Xtr, Ytr, mu, sd, n=100)

# ---------- 4. auto-box MULTI cho eval200 ----------
print("[4] build_dataset eval200 (features) + propose multi-box..."); _, _, _, _, per = build_dataset(eval200)
autob = {}
for s in eval200:
    rec = per.get(s)
    if rec is None: autob[s] = []; continue
    valid = [ci for ci, v in enumerate(rec["valid"]) if v]
    if not valid: autob[s] = []; continue
    X = np.array([rec["feat"][ci] for ci in valid], np.float32)
    vote, _ = ensemble_vote(models, X, mu, sd)
    vbc = {ci: float(v) for ci, v in zip(valid, vote)}
    autob[s] = auto_box(rec, vbc, vote_thr=VOTE_THR, multi=True) or []

# ---------- 5. SAM zero-shot & FT(e2e) với auto-box -> GỘP ----------
def seg_union(P, rgb, boxes):
    H, W = rgb.shape[:2]; u = np.zeros((H, W), bool)
    if not boxes: return u
    with torch.inference_mode(), AC:
        P.set_image(rgb)
        for b in boxes:
            mk, sc, _ = P.predict(box=np.array(b, np.float32), multimask_output=True)
            u |= mk[int(np.argmax(sc))].astype(bool)
    return u
def run(ckpt):
    m = build_sam2("configs/sam2.1_hiera_t512", "checkpoints/sam2.1_hiera_tiny.pt", device="cuda",
                   hydra_overrides_extra=[f"++model.image_size={RES}"])
    if ckpt: m.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=False)["model"], strict=False)
    P = SAM2ImagePredictor(m); out = {}
    for s in eval200:
        rgb = cv2.cvtColor(cv2.imread(f"data/20241212/{s}.jpg"), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(f"labels/masks/{s}.png", 0) > 127
        mask = seg_union(P, rgb, autob[s])
        out[s] = (dice(mask, gt), hd95(mask, gt), mask)
    del m, P; gc.collect(); torch.cuda.empty_cache(); return out
print("[5] SAM zero-shot..."); ZS = run(None)
print("[5] SAM fine-tuned(e2e)..."); FT = run(FT_CKPT)

# ---------- 6. báo cáo ----------
def nobj(s):
    p = f"labels/prompts/{s}.json"
    return json.load(open(p)).get("n_objects", 1) if os.path.isfile(p) else 1
rows = []
for s in eval200:
    rows.append((s, ZS[s][0], FT[s][0], ZS[s][1], FT[s][1], len(autob[s]), nobj(s)))
with open("results/e2e_ft_vs_zs.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["stem", "dice_zs", "dice_ft", "hd95_zs", "hd95_ft", "n_autobox", "n_obj"])
    for r in rows: w.writerow([r[0], round(r[1], 4), round(r[2], 4), round(r[3], 2), round(r[4], 2), r[5], r[6]])
Z = np.array([r[1] for r in rows]); F = np.array([r[2] for r in rows])
NB = np.array([r[5] for r in rows]); NO = np.array([r[6] for r in rows]); d = F - Z
nbox_found = (NB > 0).sum()
print(f"\n===== END-TO-END (auto multi-box -> SAM -> vs mask tay), N={len(rows)} =====")
print(f"auto-box tìm thấy >=1 box: {nbox_found}/{len(rows)} | tổng box trung vị/ảnh: {np.median(NB):.0f}")
print(f"Zero-shot  Dice median={np.median(Z):.4f} mean={np.mean(Z):.4f}")
print(f"Fine-tuned Dice median={np.median(F):.4f} mean={np.mean(F):.4f}")
print(f"delta FT-ZS mean={d.mean():+.4f} | FT thắng/thua={int((d>1e-4).sum())}/{int((d<-1e-4).sum())}")
print(f"  1 u  (n={int((NO<=1).sum())}): ZS={np.median(Z[NO<=1]):.3f} FT={np.median(F[NO<=1]):.3f}")
print(f"  >1 u (n={int((NO>1).sum())}): ZS={np.median(Z[NO>1]):.3f} FT={np.median(F[NO>1]):.3f}")
print("-> results/e2e_ft_vs_zs.csv")

# ---------- montage [GỐC | FT | ZS] sắp theo Dice ZS tăng dần ----------
order = sorted(range(len(rows)), key=lambda i: Z[i])
sub = order[:18] + order[-6:]          # 18 ca ZS kém nhất + 6 ca tốt nhất
def overlay(rgb, mask, color, gt, boxes):
    o = rgb.copy()
    if mask is not None and mask.any(): o[mask] = (0.45*o[mask] + 0.55*np.array(color)).astype(np.uint8)
    for b in boxes: cv2.rectangle(o, (b[0], b[1]), (b[2], b[3]), (60, 120, 255), 3)
    gc_, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o, gc_, -1, (255, 255, 0), 4)
    return o
PW = 340; tiles = []
for i in sub:
    s = rows[i][0]; gt = cv2.imread(f"labels/masks/{s}.png", 0) > 127
    rgb = cv2.cvtColor(cv2.imread(f"data/20241212/{s}.jpg"), cv2.COLOR_BGR2RGB)
    ys, xs = np.where(gt); m = 160
    sl = (slice(max(0, ys.min()-m), min(rgb.shape[0], ys.max()+m)),
          slice(max(0, xs.min()-m), min(rgb.shape[1], xs.max()+m)))
    def pan(img, lab, col):
        H = int(PW*img.shape[0]/img.shape[1]); im = cv2.resize(img, (PW, H))
        bar = np.full((26, PW, 3), 25, np.uint8); cv2.putText(bar, lab, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        return np.vstack([bar, im])
    p_o = pan(rgb[sl], f"GOC ({rows[i][6]}u, autobox={rows[i][5]})", (255, 255, 255))
    p_f = pan(overlay(rgb, FT[s][2], (255, 40, 40), gt, autob[s])[sl], f"FT  d={F[i]:.3f}", (120, 120, 255))
    p_z = pan(overlay(rgb, ZS[s][2], (0, 220, 0), gt, autob[s])[sl], f"ZS  d={Z[i]:.3f}", (120, 255, 120))
    h = max(p_o.shape[0], p_f.shape[0], p_z.shape[0])
    pad = lambda p: np.vstack([p, np.full((h-p.shape[0], PW, 3), 25, np.uint8)])
    sep = np.full((h, 8, 3), 60, np.uint8)
    tiles.append(np.hstack([pad(p_o), sep, pad(p_f), sep, pad(p_z)]))
COLS = 2
grows = []
for i in range(0, len(tiles), COLS):
    rw = tiles[i:i+COLS]; h = max(c.shape[0] for c in rw)
    rw = [np.vstack([c, np.full((h-c.shape[0], c.shape[1], 3), 15, np.uint8)]) for c in rw]
    while len(rw) < COLS: rw.append(np.full((h, rw[0].shape[1], 3), 15, np.uint8))
    g = np.full((h, 16, 3), 15, np.uint8); o = rw[0]
    for c in rw[1:]: o = np.hstack([o, g, c])
    grows.append(o)
W = max(r.shape[1] for r in grows)
grows = [np.hstack([r, np.full((r.shape[0], W-r.shape[1], 3), 15, np.uint8)]) for r in grows]
grid = np.vstack(grows)
leg = np.full((46, W, 3), 0, np.uint8)
cv2.putText(leg, "END-TO-END auto multi-box. Moi ca: GOC | FT(do) | ZS(xanh); vien VANG=GT, khung xanh=auto-box. 18 ca ZS kem nhat + 6 tot nhat.",
            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
grid = np.vstack([leg, grid])
cv2.imwrite("results/e2e_montage.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 4])
print("-> results/e2e_montage.png", grid.shape)
