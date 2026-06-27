"""
DELIVERABLE v2 — box→mask cho ảnh đại thể RCC bằng SEGMENTER THUẦN (SegResNet+LAB) trên CROP.
Thay SAM2 ở khâu segment: khi đưa box đúng, segmenter crop KHÔNG tô lố như SAM
→ ceiling 50 vẽ tay 0.883 > SAM 0.857 (đặc biệt đa-u 0.88 vs 0.84). KHÔNG cần SAM.

Người dùng vẽ 1 (hoặc nhiều) bounding-box quanh KHỐI U → mỗi box: crop+pad15%→resize512
→ SegResNet (6ch RGB+LAB) → mask trong patch → ghép full-res (union nhiều box).

------------------------------------------------------------------ CÁCH DÙNG
1 ảnh, 1 box:
  python predict_seg_crop.py --image a.jpg --box "x0,y0,x1,y1" --out a_mask.png
1 ảnh, nhiều box (lặp --box) + overlay QC:
  python predict_seg_crop.py --image a.jpg --box "120,80,400,350" --box "600,500,820,700" \
      --out a_mask.png --overlay a_ov.jpg
Batch CSV (cột image,x0,y0,x1,y1 ; nhiều dòng cùng image -> union):
  python predict_seg_crop.py --csv boxes.csv --out_dir out_masks [--overlay_dir out_ov]

Toạ độ box theo PIXEL ảnh GỐC. Output PNG nhị phân full-res (0=nền,255=u).
Checkpoint qua env SEG_CKPT (mặc định checkpoints/seg_crop_segR_lab.pt — champion ceiling).
Env: /home/hvusynh2/conda_envs/medsam2_anno/bin/python (monai+skimage).
"""
import argparse, csv, os, sys
import cv2, numpy as np, torch
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from seg_crop import make_channels, pad_box, build_model, SIZE

SEG_CKPT = os.environ.get("SEG_CKPT", "checkpoints/seg_crop_segR_lab.pt")
THR = float(os.environ.get("SEG_THR", "0.5"))
PAD = float(os.environ.get("SEG_PAD", "0.15"))


def load_segmenter():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(SEG_CKPT, map_location=device, weights_only=False)
    net = build_model(ck["arch"], _n_ch(ck["channels"])).to(device)
    net.load_state_dict(ck["net"]); net.eval()
    return net, ck["channels"], device


def _n_ch(mode):
    from seg_crop import n_channels
    return n_channels(mode)


@torch.no_grad()
def boxes_to_mask(net, mode, device, bgr, boxes):
    """bgr HxWx3 uint8, boxes list[x0,y0,x1,y1] -> mask bool HxW (union)."""
    H, W = bgr.shape[:2]; union = np.zeros((H, W), bool)
    for b in boxes:
        x0, y0, x1, y1 = pad_box(list(map(float, b)), PAD, W, H)
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        cr = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(make_channels(np.ascontiguousarray(cr), mode))[None].to(device)
        pm = torch.sigmoid(net(x))[0, 0].float().cpu().numpy()
        pm = cv2.resize(pm, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > THR
        union[y0:y1, x0:x1] |= pm
    return union


def clip_box(box, W, H):
    x0, y0, x1, y1 = box
    x0, x1 = sorted((max(0, min(W - 1, int(x0))), max(0, min(W, int(x1)))))
    y0, y1 = sorted((max(0, min(H - 1, int(y0))), max(0, min(H, int(y1)))))
    return [x0, y0, x1, y1]


def make_overlay(bgr, mask, boxes):
    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
    out[mask] = (0.5 * out[mask] + 0.5 * np.array([0, 255, 120])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 0), 4)
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(out, (x0, y0), (x1, y1), (60, 120, 255), 4)
    return out


def parse_box(s):
    parts = [float(v) for v in s.replace(";", ",").replace(" ", ",").split(",") if v != ""]
    if len(parts) != 4:
        raise ValueError(f"box phải có 4 số x0,y0,x1,y1 — nhận: {s!r}")
    return parts


def run_one(net, mode, device, img_path, boxes, out_path, overlay_path=None):
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"[bỏ] không đọc được ảnh: {img_path}"); return False
    H, W = bgr.shape[:2]
    boxes = [clip_box(b, W, H) for b in boxes]
    mask = boxes_to_mask(net, mode, device, bgr, boxes)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, (mask.astype(np.uint8) * 255))
    if overlay_path:
        os.makedirs(os.path.dirname(os.path.abspath(overlay_path)), exist_ok=True)
        ov = make_overlay(bgr, mask, boxes)
        cv2.imwrite(overlay_path, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"[ok] {os.path.basename(img_path)} | {len(boxes)} box | "
          f"u={int(mask.sum())}px ({100*mask.sum()/(H*W):.1f}% ảnh) -> {out_path}")
    return True


def read_csv_boxes(path):
    by_img = {}
    for r in csv.DictReader(open(path)):
        img = r["image"].strip()
        by_img.setdefault(img, []).append([float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])])
    return by_img


def resolve_image(img):
    for cand in (img, os.path.join(ROOT, img), os.path.join(ROOT, "data/20241212", img),
                 os.path.join(ROOT, "data/20241212", img + ".jpg")):
        if os.path.isfile(cand):
            return cand
    return img


def main():
    ap = argparse.ArgumentParser(description="SegResNet+LAB box->mask trên crop (RCC gross)")
    ap.add_argument("--image"); ap.add_argument("--box", action="append", default=[])
    ap.add_argument("--out"); ap.add_argument("--overlay")
    ap.add_argument("--csv"); ap.add_argument("--out_dir"); ap.add_argument("--overlay_dir")
    args = ap.parse_args()
    net, mode, device = load_segmenter()
    print(f"segmenter={SEG_CKPT} channels={mode} pad={PAD} thr={THR} device={device}")
    if args.csv:
        if not args.out_dir: ap.error("--csv cần kèm --out_dir")
        by_img = read_csv_boxes(args.csv); print(f"Batch: {len(by_img)} ảnh từ {args.csv}")
        n = 0
        for img, boxes in by_img.items():
            ip = resolve_image(img); stem = os.path.splitext(os.path.basename(ip))[0]
            ovp = os.path.join(args.overlay_dir, stem + ".jpg") if args.overlay_dir else None
            n += run_one(net, mode, device, ip, boxes, os.path.join(args.out_dir, stem + ".png"), ovp)
        print(f"Xong: {n}/{len(by_img)} ảnh -> {args.out_dir}")
    else:
        if not (args.image and args.box and args.out):
            ap.error("chế độ 1 ảnh cần --image, ít nhất 1 --box, và --out")
        run_one(net, mode, device, resolve_image(args.image),
                [parse_box(b) for b in args.box], args.out, args.overlay)


if __name__ == "__main__":
    main()
