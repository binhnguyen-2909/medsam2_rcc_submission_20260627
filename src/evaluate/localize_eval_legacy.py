"""
HARNESS EVAL LOCALIZE dùng chung — mỗi localizer nhả BOX -> segmenter champion (segR_lab)
-> mask union -> Dice trên 50 ảnh vẽ tay. Cô lập CHẤT LƯỢNG LOCALIZE (segmenter cố định).
So full-auto detector 0.635 / ceiling 0.883. Mỗi method lưu results/loc_<method>.json.

  python localize_eval.py --method detector|refiner|iter|yolo|slic|grid|centerpoint|gdino

centerpoint: detector center -> SAM point_coords (không qua segmenter, đo nhánh point-prompt).
gdino: Grounding DINO text-prompt (nếu cài được). Env: medsam2_anno.
"""
import argparse, json, os, sys, csv, numpy as np, cv2, torch, statistics as st
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import predict_seg_crop as P
from seg_crop import frag_boxes, ncomp, pad_box, make_channels, SIZE, pat
from detector import DenseDetector, propose_boxes
from specimen_clean import clean_specimen
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
AC = torch.autocast("cuda", dtype=torch.bfloat16); DEVICE = "cuda"

def dice(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s

_pred = _det = _seg = _segmode = None
def sam_pred():
    global _pred
    if _pred is None:
        m = build_sam2("configs/sam2.1_hiera_t512", "checkpoints/sam2.1_hiera_tiny.pt",
                       device=DEVICE, hydra_overrides_extra=["++model.image_size=1024"])
        _pred = SAM2ImagePredictor(m)
    return _pred
def detector():
    global _det
    if _det is None:
        dk = torch.load("checkpoints/detector_recall.pt", weights_only=False)
        _det = DenseDetector(grid=dk.get("grid", 64)).to(DEVICE); _det.load_state_dict(dk["det"]); _det.eval()
    return _det
def segmenter():
    global _seg, _segmode
    if _seg is None: _seg, _segmode, _ = P.load_segmenter()
    return _seg, _segmode

def detector_boxes(bgr, rgb, spec):
    H, W = bgr.shape[:2]; pr = sam_pred()
    with torch.inference_mode(), AC:
        pr.set_image(rgb); feat = pr._features["image_embed"].float(); obj, bx = detector()(feat)
    return list(propose_boxes(obj[0].float(), bx[0].float(), H, W, spec=spec, thr=0.5))

def bbox(m):
    ys, xs = np.where(m); return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())] if len(ys) else None

def seg_mask(bgr, boxes):
    net, mode = segmenter()
    return P.boxes_to_mask(net, mode, DEVICE, bgr, boxes) if boxes else np.zeros(bgr.shape[:2], bool)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); args = ap.parse_args()
    method = args.method
    stems = json.load(open("labels_handdraw/select.json"))["stems"]
    have = [s for s in stems if os.path.isfile(f"labels_handdraw/masks/{s}.png")]

    # khởi tạo localizer phụ thuộc method
    refiner = yolo = slic_m = grid_m = gdino = amg_m = ae_m = sizemap = None
    if method == "refiner":
        from box_refiner import load_refiner, refine_box; refiner = (load_refiner(), refine_box)
    elif method == "yolo":
        from ultralytics import YOLO; yolo = YOLO("checkpoints/yolo_best.pt")
    elif method == "slic":
        import slic_clf; slic_m = (slic_clf, slic_clf.load_clf())
    elif method == "grid":
        import grid_clf; grid_m = (grid_clf, grid_clf.load_grid())
    elif method == "amg":
        import amg_classify; amg_m = (amg_classify, amg_classify.load_clf())
    elif method == "anomaly":
        import anomaly_ae; ae_m = (anomaly_ae, anomaly_ae.load_ae())
    elif method == "size":
        sizemap = load_sizemap()
    elif method == "gdino":
        gdino = load_gdino()
        if gdino is None: print("[gdino] không tải được — bỏ qua."); return

    rows = []
    for s in have:
        gt = cv2.imread(f"labels_handdraw/masks/{s}.png", 0) > 127
        bgr = cv2.imread(f"data/20241212/{s}.jpg"); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = gt.shape; spec, _, _ = clean_specimen(bgr); no = ncomp(gt)
        if method == "detector":
            mk = seg_mask(bgr, detector_boxes(bgr, rgb, spec))
        elif method == "refiner":
            net, rfn = refiner; ab = detector_boxes(bgr, rgb, spec)
            rb = [rfn(net, bgr, b) for b in ab]; mk = seg_mask(bgr, rb)
        elif method == "iter":
            ab = detector_boxes(bgr, rgb, spec); net, mode = segmenter(); union = np.zeros((H, W), bool)
            for b in ab:
                m0 = P.boxes_to_mask(net, mode, DEVICE, bgr, [b]); bb = bbox(m0)
                m1 = P.boxes_to_mask(net, mode, DEVICE, bgr, [bb]) if bb else m0
                union |= m1
            mk = union
        elif method == "yolo":
            r = yolo.predict(bgr, imgsz=1024, conf=0.25, verbose=False, device=0)[0]
            boxes = [list(map(float, b)) for b in r.boxes.xyxy.cpu().numpy()] if r.boxes is not None else []
            # gate specimen
            boxes = [b for b in boxes if spec[int(np.clip((b[1]+b[3])/2,0,H-1)), int(np.clip((b[0]+b[2])/2,0,W-1))] > 0] or boxes
            mk = seg_mask(bgr, boxes)
        elif method == "slic":
            mod, (net, mean, std, thr) = slic_m
            mk = seg_mask(bgr, mod.slic_boxes(bgr, spec, net, mean, std))
        elif method == "grid":
            mod, net = grid_m; mk = seg_mask(bgr, mod.grid_boxes(bgr, spec, net))
        elif method == "centerpoint":
            ab = detector_boxes(bgr, rgb, spec); pr = sam_pred(); union = np.zeros((H, W), bool)
            with torch.inference_mode(), AC:
                pr.set_image(rgb)
                for b in ab:
                    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
                    pts = np.array([[cx, cy]], np.float32); lab = np.array([1], np.int32)
                    msk, sc, _ = pr.predict(point_coords=pts, point_labels=lab, multimask_output=True)
                    union |= msk[int(np.argmax(sc))].astype(bool)
            mk = union
        elif method == "amg":
            mod, net = amg_m; mk = seg_mask(bgr, mod.amg_boxes(sam_pred(), net, rgb, bgr, spec))
        elif method == "anomaly":
            mod, net = ae_m; mk = seg_mask(bgr, mod.anomaly_boxes(net, bgr, spec))
        elif method == "size":
            ab = detector_boxes(bgr, rgb, spec)
            fb = size_filter(ab, s, spec, sizemap)
            mk = seg_mask(bgr, fb)
        elif method == "gdino":
            boxes = gdino_boxes(gdino, rgb, spec)
            mk = seg_mask(bgr, boxes)
        rows.append((no, dice(mk, gt)))
        print(f"  {s[:20]:22} n={no} dice={rows[-1][1]:.3f}", flush=True)

    def agg(f=lambda r: True):
        v = [r[1] for r in rows if f(r)]; return (st.median(v), st.mean(v), len(v)) if v else (0, 0, 0)
    A, A1, Am = agg(), agg(lambda r: r[0] <= 1), agg(lambda r: r[0] > 1)
    print(f"\n===== LOCALIZE [{method}] -> segmenter -> 50 vẽ tay =====")
    print(f"Dice median={A[0]:.4f} mean={A[1]:.4f} (n={A[2]}) | 1u={A1[0]:.3f}(n{A1[2]}) >1u={Am[0]:.3f}(n{Am[2]})")
    print(f"SO: full-auto detector=0.635 | ceiling box-GT=0.883 | nhãn SAM cũ=0.554", flush=True)
    json.dump({"method": method, "median": A[0], "mean": A[1], "1u": A1[0], ">1u": Am[0], "n": A[2]},
              open(f"results/loc_{method}.json", "w"), indent=1)
    print(f"-> results/loc_{method}.json", flush=True)

# ---- #3 ràng buộc kích thước từ Excel ----
def load_sizemap():
    """canon -> mass_area_cm2 ; stem -> px_per_cm (từ Excel + metadata)."""
    area = {}
    if os.path.isfile("processed/excel_parsed.csv"):
        for r in csv.DictReader(open("processed/excel_parsed.csv")):
            try: area[r["canon"]] = float(r["mass_area_cm2"])
            except (ValueError, KeyError): pass
    ppc = {}
    if os.path.isfile("processed/metadata.csv"):
        for r in csv.DictReader(open("processed/metadata.csv")):
            try:
                v = float(r.get("px_per_cm", "") or "nan")
                if v == v and v > 0: ppc[r.get("stem", "")] = v
            except ValueError: pass
    return {"area": area, "ppc": ppc}

def size_filter(boxes, stem, spec, sizemap):
    """Giữ box có diện tích vật lý (cm2) hợp lý so với Excel. Thiếu px/cm hoặc Excel -> giữ hết."""
    canon = pat(stem).replace("-", "")
    a_cm2 = sizemap["area"].get(canon); ppc = sizemap["ppc"].get(stem)
    if not a_cm2 or not ppc:
        return boxes
    out = []
    for b in boxes:
        area_px = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        cm2 = area_px / (ppc * ppc)
        if 0.15 * a_cm2 <= cm2 <= 6.0 * a_cm2:   # ràng buộc lỏng: triệt box vô lý
            out.append(b)
    return out or boxes   # nếu loại hết thì giữ nguyên (an toàn)

# ---- Grounding DINO (tùy chọn, nếu cài được) ----
def load_gdino():
    try:
        from groundingdino.util.inference import load_model, predict  # noqa
        import groundingdino
        cfgs = [f for f in os.listdir(".") if "GroundingDINO" in f]
        cfg = "GroundingDINO_SwinT_OGC.py"; wt = "checkpoints/groundingdino_swint_ogc.pth"
        if not os.path.isfile(wt): return None
        return load_model(cfg, wt)
    except Exception as e:
        print(f"[gdino] {e}"); return None

def gdino_boxes(model, rgb, spec):
    from groundingdino.util.inference import predict
    import groundingdino.datasets.transforms as T
    from PIL import Image
    H, W = rgb.shape[:2]
    tf = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img, _ = tf(Image.fromarray(rgb), None)
    boxes, logits, phrases = predict(model=model, image=img,
        caption="renal cell carcinoma mass, distinct abnormal tissue texture",
        box_threshold=0.25, text_threshold=0.25)
    out = []
    for b in boxes.cpu().numpy():
        cx, cy, bw, bh = b; x0 = (cx - bw/2) * W; y0 = (cy - bh/2) * H; x1 = (cx + bw/2) * W; y1 = (cy + bh/2) * H
        out.append([x0, y0, x1, y1])
    return out

if __name__ == "__main__":
    main()
