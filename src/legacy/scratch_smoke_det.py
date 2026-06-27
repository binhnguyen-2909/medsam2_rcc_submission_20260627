"""Smoke test dense detector: forward -> assign -> loss -> decode_detections -> mask-loss -> backward -> val."""
import os, sys, json, torch, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import train_detector as T
from detector import DenseDetector, assign_targets, dense_losses, decode_detections
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

done = [s for s in json.load(open("labels/done.json")) if os.path.isfile(f"labels/masks/{s}.png")]
multi, single = [], []
for s in done:
    g = T.gt_targets(s)
    if g is None: continue
    (multi if len(g[0]) > 1 else single).append(s)
    if len(multi) >= 2 and len(single) >= 2: break
stems = multi[:2] + single[:2]
print("stems:", [(s[:10], len(T.gt_targets(s)[0])) for s in stems])

model = build_sam2(T.CFG, T.CKPT, device="cuda", hydra_overrides_extra=["++model.image_size=1024"])
for p in model.parameters(): p.requires_grad_(False)
model.eval()
pred = SAM2ImagePredictor(model); tr = pred._transforms
T.precompute(pred, stems, [False])

det = DenseDetector().to("cuda")
opt = torch.optim.AdamW(det.parameters(), lr=1e-4)
for step in range(3):
    feats = [torch.load(T.cache_path(s, False), weights_only=False)["feat"].float() for s in stems]
    cks = [torch.load(T.cache_path(s, False), weights_only=False) for s in stems]
    feat = torch.stack(feats).to("cuda")
    with T.AC:
        obj, boxes = det(feat)
    obj = obj.float(); boxes = boxes.float()
    loss = 0.0
    for bi, ck in enumerate(cks):
        gtb = torch.as_tensor(ck["boxes"], device="cuda")
        pos, tgt, gidx = assign_targets(det.centers, gtb)
        dl = dense_losses(obj[bi], boxes[bi], pos, tgt)
        l = dl["cls"] + 5 * dl["l1"] + 2 * dl["giou"]
        # mask loss (1 ô/GT)
        gi_pos = gidx[pos]; pb = boxes[bi][pos]; sel, segb = [], []
        for g in gi_pos.unique():
            idxs = (gi_pos == g).nonzero().squeeze(1); sel.append(pb[idxs[len(idxs)//2]]); segb.append(int(g))
        pbs = torch.stack(sel)
        hrf = [h.float().to("cuda") for h in ck["hrf"]]
        comps = torch.as_tensor(ck["comps"][np.array(segb)], device="cuda").float()
        with T.AC:
            ml = T.decode_masks(model, tr, feat[bi], hrf, ck["hw"], pbs)
        mloss = T.dice_bce_loss(ml.float(), comps)
        l = l + mloss; loss = loss + l
        if step == 0:
            bxk, sc = decode_detections(obj[bi], boxes[bi], thr=0.3)
            print(f"  img{bi}: pos_cells={int(pos.sum())} gt={len(gtb)} cls={float(dl['cls']):.3f} "
                  f"l1={float(dl['l1']):.3f} giou={float(dl['giou']):.3f} mask={float(mloss):.3f} "
                  f"decode_boxes={len(bxk)}")
    loss = loss / len(cks)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(det.parameters(), 1.0); opt.step()
    print(f"step{step} loss={float(loss):.3f}")

vd, vt = T.val_dice(det, model, tr, stems[:2])
print(f"val_dice={vd:.4f}@thr{vt}")
print("SMOKE OK")
