"""
Tool GÁN NHÃN KHỐI U trên ảnh đại thể RCC (deliverable box->mask) bằng SAM2.1@1024.

ĐỐI TƯỢNG = CHỈ KHỐI U (chừa mô thận lành/vỏ). KHÔNG khoanh cả lát bệnh phẩm.
Một ảnh có thể có NHIỀU mảnh/nhiều u -> gán từng u rồi gộp (union) khi lưu.

Quy trình 1 ảnh:
  1. Mode "Box": click 2 góc quanh RIÊNG khối u -> SAM2.1 sinh mask.
  2. Tinh chỉnh: "+điểm" (thêm vùng u) / "-điểm" (bỏ phần lấn mô lành) -> chạy lại.
  3. "➕ Thêm u" để lưu tạm u này rồi khoanh u/mảnh kế (mask sẽ cộng dồn).
  4. "Accept & Lưu" -> gộp tất cả u thành 1 mask nhị phân, sang ảnh kế.
     "Skip" (ảnh mặt ngoài/không có u) | "Bỏ u đã thêm" | "◀ Prev / Next ▶".

Quay lại ảnh đã làm (Prev/Next) sẽ NẠP LẠI mask cũ để soi/sửa.

Lưu vào labels/:
  masks/<stem>.png      union mask nhị phân full-res (0/255)
  overlays/<stem>.jpg   overlay QC
  prompts/<stem>.json   list instance {box,điểm,score,area} (tái lập / train)
  annotations.csv       1 dòng/ảnh (n_objects, tổng area, kích u Excel)
  done.json, skipped.json  trạng thái RESUME

Chạy: /home/hvusynh2/conda_envs/medsam2_anno/bin/python annotate.py  (cổng 18863)
"""
import csv
import json
import os
import sys
from glob import glob

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

IMG_DIR = os.path.join(ROOT, "data/20241212")
LABELS_DIR = os.path.join(ROOT, "labels")
CONFIG = os.environ.get("SAM2_CONFIG", "configs/sam2.1_hiera_t512")
CKPT = os.environ.get("SAM2_CKPT", "checkpoints/sam2.1_hiera_tiny.pt")
RES = int(os.environ.get("SAM2_RES", "1024"))
EXCEL_CSV = os.path.join(ROOT, "processed/excel_parsed.csv")
PORT = int(os.environ.get("ANNO_PORT", "18863"))

# Chế độ REVIEW: chỉ nạp các ảnh có CỜ trong CSV (cột "flags" khác rỗng) để SỬA LẠI.
# Bật bằng:  ANNO_REVIEW=labels/relabel_queue.csv python annotate.py
REVIEW_CSV = os.environ.get("ANNO_REVIEW", "").strip()
# Chế độ QUEUE: chỉ duyệt ảnh MẶT-CẮT (processed/cut_surface_filter.csv),
# bỏ ảnh mặt-ngoài. Bật bằng: ANNO_QUEUE=cut python annotate.py
QUEUE_CUT = os.environ.get("ANNO_QUEUE", "").strip().lower() == "cut"

for sub in ("masks", "overlays", "prompts"):
    os.makedirs(os.path.join(LABELS_DIR, sub), exist_ok=True)


def review_stems():
    """Đọc danh sách stem có cờ từ CSV review (giữ nguyên thứ tự trong file)."""
    p = REVIEW_CSV if os.path.isabs(REVIEW_CSV) else os.path.join(ROOT, REVIEW_CSV)
    stems = []
    if os.path.isfile(p):
        for r in csv.DictReader(open(p)):
            if (r.get("flags") or "").strip():
                stems.append(r["stem"])
    return stems


# ----------------------------- helpers I/O -----------------------------
def cut_surface_stems():
    """Stem ảnh MẶT-CẮT (is_cut_surface=1) từ processed/cut_surface_filter.csv."""
    p = os.path.join(ROOT, "processed/cut_surface_filter.csv")
    out = []
    if os.path.isfile(p):
        for r in csv.DictReader(open(p)):
            if r.get("is_cut_surface") == "1":
                out.append(r["stem"])
    return out


def list_images():
    if REVIEW_CSV:
        out = []
        for s in review_stems():
            p = os.path.join(IMG_DIR, s + ".jpg")
            if os.path.isfile(p):
                out.append(p)
            else:
                print("  [review] thiếu ảnh:", s)
        print(f"[REVIEW] nạp {len(out)} ảnh có cờ từ {REVIEW_CSV}")
        return out
    if QUEUE_CUT:
        # ưu tiên thứ tự active-learning nếu có labels/queue_order.json
        order_p = os.path.join(ROOT, "labels/queue_order.json")
        if os.path.isfile(order_p):
            stems = json.load(open(order_p))
            out = [os.path.join(IMG_DIR, s + ".jpg") for s in stems
                   if os.path.isfile(os.path.join(IMG_DIR, s + ".jpg"))]
            print(f"[QUEUE] nạp {len(out)} ảnh MẶT-CẮT theo THỨ TỰ ƯU TIÊN")
            return out
        out = sorted(os.path.join(IMG_DIR, s + ".jpg") for s in cut_surface_stems()
                     if os.path.isfile(os.path.join(IMG_DIR, s + ".jpg")))
        print(f"[QUEUE] nạp {len(out)} ảnh MẶT-CẮT (bỏ mặt-ngoài)")
        return out
    return sorted(glob(os.path.join(IMG_DIR, "*.jpg")))


def stem_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def patient_of(stem):
    return stem.split("^")[0]


def canon_of(stem):
    return patient_of(stem).replace("-", "")


def load_excel_mass():
    m = {}
    if os.path.isfile(EXCEL_CSV):
        for r in csv.DictReader(open(EXCEL_CSV)):
            m[r["canon"]] = r
    return m


def load_set(name):
    p = os.path.join(LABELS_DIR, name)
    if os.path.isfile(p):
        try:
            return set(json.load(open(p)))
        except Exception:
            return set()
    return set()


def save_set(name, s):
    json.dump(sorted(s), open(os.path.join(LABELS_DIR, name), "w"))


EXCEL = load_excel_mass()


# ----------------------------- model -----------------------------
print("Nạp SAM2.1 @%d ..." % RES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model = build_sam2(CONFIG, CKPT, device=DEVICE,
                    hydra_overrides_extra=[f"++model.image_size={RES}"])
PREDICTOR = SAM2ImagePredictor(_model)
_AC = (torch.autocast("cuda", dtype=torch.bfloat16) if DEVICE == "cuda"
       else torch.autocast("cpu", enabled=False))
print(f"Model sẵn sàng trên {DEVICE}.")

# Auto-box proposer (ensemble ô-lưới) -> prefill box cho ảnh CHƯA gán.
# Không nạp được thì app vẫn chạy bình thường (chỉ mất phần gợi ý box).
try:
    from propose_box import load_proposer, propose as _propose_box
    PROPOSER = load_proposer()
    print("Auto-box sẵn sàng (%d model)." % len(PROPOSER["models"]))
except Exception as e:
    PROPOSER = None
    print("Auto-box KHÔNG nạp được (%s) — bỏ phần prefill box." % e)

_CUR = {"stem": None}


def ensure_image(rgb, stem):
    if _CUR["stem"] != stem:
        with torch.inference_mode(), _AC:
            PREDICTOR.set_image(rgb)
        _CUR["stem"] = stem


def run_sam(rgb, stem, box, pos_pts, neg_pts):
    ensure_image(rgb, stem)
    pc, pl = None, None
    if pos_pts or neg_pts:
        pts = pos_pts + neg_pts
        labs = [1] * len(pos_pts) + [0] * len(neg_pts)
        pc = np.array(pts, dtype=np.float32)
        pl = np.array(labs, dtype=np.int32)
    bx = np.array(box, dtype=np.float32) if box else None
    multimask = (pc is None)
    with torch.inference_mode(), _AC:
        masks, scores, _ = PREDICTOR.predict(
            point_coords=pc, point_labels=pl, box=bx, multimask_output=multimask)
    bi = int(np.argmax(scores))
    return masks[bi].astype(bool), float(scores[bi])


_REF = {"mask": None}  # review mode: mask CŨ chỉ vẽ viền đỏ tham chiếu, KHÔNG lưu


def overlay_np(rgb, committed, cur_mask, box, pos_pts, neg_pts):
    """committed = list mask đã thêm (xanh đậm); cur_mask = đang làm (xanh sáng).
    _REF['mask'] (nếu có) = mask cũ cần sửa -> vẽ viền đỏ mảnh để đối chiếu."""
    out = rgb.copy()
    if _REF["mask"] is not None:
        cnts, _ = cv2.findContours(_REF["mask"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 40, 40), 2)
    for m in committed:
        out[m] = (0.5 * out[m] + 0.5 * np.array([0, 150, 60])).astype(np.uint8)
    if cur_mask is not None:
        out[cur_mask] = (0.45 * out[cur_mask] + 0.55 * np.array([0, 255, 120])).astype(np.uint8)
    allm = None
    for m in committed + ([cur_mask] if cur_mask is not None else []):
        allm = m if allm is None else (allm | m)
    if allm is not None:
        cnts, _ = cv2.findContours(allm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 0), 4)
    if box:
        boxes = box if isinstance(box[0], (list, tuple)) else [box]   # đơn hoặc nhiều box
        for bx in boxes:
            x0, y0, x1, y1 = [int(v) for v in bx]
            cv2.rectangle(out, (x0, y0), (x1, y1), (60, 120, 255), 4)
    r = max(6, int(min(rgb.shape[:2]) * 0.006))
    for x, y in pos_pts:
        cv2.circle(out, (int(x), int(y)), r, (0, 255, 0), -1)
        cv2.circle(out, (int(x), int(y)), r, (255, 255, 255), 2)
    for x, y in neg_pts:
        cv2.circle(out, (int(x), int(y)), r, (255, 0, 0), -1)
        cv2.circle(out, (int(x), int(y)), r, (255, 255, 255), 2)
    return out


def mass_info(stem):
    c = canon_of(stem)
    r = EXCEL.get(c)
    if not r:
        return "Excel: (không khớp khoá %s)" % c
    return (f"Excel mass: {r.get('mass_dims','?')}cm ~{r.get('mass_area_cm2','?')}cm² "
            f"(n_mass={r.get('n_mass','?')}, {r.get('parse_flag','?')})")


# ----------------------------- Gradio app -----------------------------
def build_app():
    import gradio as gr

    IMAGES = list_images()
    done = load_set("done.json")
    skipped = load_set("skipped.json")
    # review mode: theo dõi ảnh đã SỬA LẠI trong lượt này (24 ảnh cờ vốn đã ở done.json)
    relabel_done = load_set("relabel_done.json") if REVIEW_CSV else set()

    def first_pending(idx_from=0):
        for i in range(idx_from, len(IMAGES)):
            s = stem_of(IMAGES[i])
            if REVIEW_CSV:
                if s not in relabel_done:   # mở ảnh chưa-sửa-lại kế tiếp
                    return i
            elif s not in done and s not in skipped:
                return i
        return min(idx_from, len(IMAGES) - 1)

    def read_rgb(i):
        return cv2.cvtColor(cv2.imread(IMAGES[i]), cv2.COLOR_BGR2RGB)

    def saved_mask(i):
        p = os.path.join(LABELS_DIR, "masks", stem_of(IMAGES[i]) + ".png")
        if os.path.isfile(p):
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                return m > 127
        return None

    _BOX_CACHE = {}

    def propose_box_for(i):
        """LIST box auto (mọi khối u) cho ảnh i: [[x0,y0,x1,y1],...] hoặc None."""
        if PROPOSER is None:
            return None
        stem = stem_of(IMAGES[i])
        if stem not in _BOX_CACHE:
            try:
                boxes = _propose_box(PROPOSER, cv2.imread(IMAGES[i]), multi=True)
                _BOX_CACHE[stem] = [[int(v) for v in b] for b in boxes] if boxes else None
            except Exception as e:
                print("[auto-box] lỗi", stem, e)
                _BOX_CACHE[stem] = None
        return _BOX_CACHE[stem]

    def status(i, committed, extra=""):
        s = stem_of(IMAGES[i])
        nobj = len(committed)
        if REVIEW_CSV:
            tag = "🔧đã-sửa" if s in relabel_done else "•cần-sửa"
            head = (f"[REVIEW {i+1}/{len(IMAGES)}] {tag} | đã sửa: "
                    f"{len(relabel_done)}/{len(IMAGES)}")
        else:
            tag = "✅done" if s in done else ("⏭skip" if s in skipped else "•chưa")
            head = (f"[{i+1}/{len(IMAGES)}] {tag} | accept:{len(done)} skip:{len(skipped)}")
        return (f"{head} | u đã thêm ảnh này: {nobj}\n{s}\n{mass_info(s)}{extra}")

    def render(i, committed, cur=None, box=None, pos=None, neg=None, sc=None):
        pos, neg = pos or [], neg or []
        rgb = read_rgb(i)
        disp = overlay_np(rgb, committed, cur, box, pos, neg)
        extra = f"\nSAM score={sc:.3f}" if sc is not None else ""
        if cur is not None:
            extra += f" | area u này={int(cur.sum())}px"
        return disp, status(i, committed, extra)

    # ---- on land: nạp mask cũ (nếu có) làm 1 committed để soi/sửa ----
    def land(i):
        committed = []
        sm = saved_mask(i)
        box = None
        if sm is not None:
            committed = [sm]          # ảnh đã gán -> nạp mask cũ, KHÔNG prefill box
        else:
            box = propose_box_for(i)  # ảnh chưa gán -> gợi ý box auto (xanh dương)
        disp, st = render(i, committed, box=box)
        if box:
            nb = len(box) if isinstance(box[0], (list, tuple)) else 1
            st += (f"\n🤖 auto-box: {nb} khối u (khung xanh dương) — KIỂM trước: bấm "
                   "'Box OK → tạo mask' để gộp tất cả; lệch thì tự vẽ Box (2 click). "
                   "(box auto thường LỎNG → siết cho sát u, tránh dính mô lành)")
        return i, committed, None, box, [], [], None, disp, st

    def on_load():
        return land(first_pending(0))

    def click(i, mode, committed, box, corner, pos, neg, evt: gr.SelectData):
        x, y = float(evt.index[0]), float(evt.index[1])
        rgb = read_rgb(i)
        stem = stem_of(IMAGES[i])
        if box and isinstance(box[0], (list, tuple)):
            box = None        # đang hiển thị nhiều auto-box -> user tương tác tay thì bỏ
        if mode == "Box (2 click)":
            if corner is None:
                corner = [x, y]
                disp = overlay_np(rgb, committed, None, None, [], [])
                cv2.drawMarker(disp, (int(x), int(y)), (60, 120, 255),
                               cv2.MARKER_CROSS, 30, 3)
                return committed, box, corner, pos, neg, None, None, disp, \
                    status(i, committed, "\ngóc 1 đã đặt — click góc 2 quanh KHỐI U")
            x0, y0 = corner
            box = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
            corner = None
            mask, sc = run_sam(rgb, stem, box, pos, neg)
            disp, st = render(i, committed, mask, box, pos, neg, sc)
            return committed, box, corner, pos, neg, mask, sc, disp, st
        if mode == "+điểm":
            pos = pos + [[x, y]]
        else:
            neg = neg + [[x, y]]
        if box is None and not pos:
            disp = overlay_np(rgb, committed, None, None, pos, neg)
            return committed, box, corner, pos, neg, None, None, disp, \
                status(i, committed, "\ncần Box hoặc +điểm trước")
        mask, sc = run_sam(rgb, stem, box, pos, neg)
        disp, st = render(i, committed, mask, box, pos, neg, sc)
        return committed, box, corner, pos, neg, mask, sc, disp, st

    def run_autobox(i, committed, box, pos, neg):
        """Chạy SAM2 cho MỌI auto-box (mỗi khối u 1 box) -> GỘP mask."""
        if box and isinstance(box[0], (list, tuple)):
            boxes = box                       # list nhiều box (auto)
        elif box:
            boxes = [box]                     # 1 box (manual đang giữ)
        else:
            boxes = propose_box_for(i)        # tự đề xuất
        if not boxes:
            disp, st = render(i, committed)
            return committed, None, None, pos, neg, None, None, disp, \
                st + "\n⚠ không đề xuất được box — hãy tự vẽ Box"
        rgb = read_rgb(i); stem = stem_of(IMAGES[i])
        union, scs = None, []
        for b in boxes:
            m, sc = run_sam(rgb, stem, b, [], [])      # từng khối u, rồi gộp
            union = m if union is None else (union | m); scs.append(sc)
        sc = float(np.mean(scs)) if scs else None
        disp, st = render(i, committed, union, boxes, pos, neg, sc)
        return committed, None, None, pos, neg, union, sc, disp, \
            st + f"\n🤖 đã gộp {len(boxes)} khối u — kiểm; +/−điểm để chỉnh, hoặc ➕Thêm u/Accept"

    def add_obj(i, committed, cur):
        if cur is not None:
            committed = committed + [cur]
        disp, st = render(i, committed)  # cur về None, prompt reset
        return committed, None, None, [], [], None, disp, st + "\n➕ đã thêm u — khoanh u/mảnh kế"

    def clear_cur(i, committed):
        disp, st = render(i, committed)
        return None, None, [], [], None, disp, st + "\nđã xoá prompt hiện tại"

    def clear_all(i):
        disp, st = render(i, [])
        return [], None, None, [], [], None, disp, st + "\nđã bỏ hết u đã thêm"

    def goto(i, j):
        return land(max(0, min(len(IMAGES) - 1, j)))

    def accept(i, committed, cur, box, pos, neg, sc):
        stem = stem_of(IMAGES[i])
        objs = list(committed) + ([cur] if cur is not None else [])
        if not objs:
            disp, st = render(i, committed)
            return (i, committed, box, None, pos, neg, cur, disp,
                    st + "\n⚠ chưa có u nào để lưu")
        rgb = read_rgb(i)
        union = np.zeros(rgb.shape[:2], bool)
        for m in objs:
            union |= m
        cv2.imwrite(os.path.join(LABELS_DIR, "masks", stem + ".png"),
                    (union.astype(np.uint8) * 255))
        ov = overlay_np(rgb, objs, None, None, [], [])
        cv2.imwrite(os.path.join(LABELS_DIR, "overlays", stem + ".jpg"),
                    cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        inst = [{"area_px": int(m.sum())} for m in objs]
        if cur is not None:
            inst[-1].update({"box": box, "pos_pts": pos, "neg_pts": neg, "score": sc})
        json.dump({"stem": stem, "n_objects": len(objs), "instances": inst,
                   "union_area_px": int(union.sum()),
                   "W": rgb.shape[1], "H": rgb.shape[0]},
                  open(os.path.join(LABELS_DIR, "prompts", stem + ".json"), "w"))
        c = canon_of(stem)
        er = EXCEL.get(c, {})
        row = {"stem": stem, "patient_id": patient_of(stem),
               "n_objects": len(objs), "union_area_px": int(union.sum()),
               "last_box": ("|".join(str(int(v)) for v in box)
                            if (box and not isinstance(box[0], (list, tuple))) else ""),
               "last_score": round(sc, 4) if sc is not None else "",
               "W": rgb.shape[1], "H": rgb.shape[0],
               "mass_dims_cm": er.get("mass_dims", ""),
               "mass_area_cm2": er.get("mass_area_cm2", "")}
        csv_p = os.path.join(LABELS_DIR, "annotations.csv")
        rows = []
        if os.path.isfile(csv_p):
            rows = [r for r in csv.DictReader(open(csv_p)) if r["stem"] != stem]
        with open(csv_p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader(); w.writerows(rows); w.writerow(row)
        done.add(stem); skipped.discard(stem)
        save_set("done.json", done); save_set("skipped.json", skipped)
        if REVIEW_CSV:
            relabel_done.add(stem); save_set("relabel_done.json", relabel_done)
        return land(first_pending(i + 1))

    def skip(i):
        stem = stem_of(IMAGES[i])
        skipped.add(stem); done.discard(stem)
        save_set("skipped.json", skipped); save_set("done.json", done)
        if REVIEW_CSV:
            relabel_done.add(stem); save_set("relabel_done.json", relabel_done)
        return land(first_pending(i + 1))

    with gr.Blocks(title="RCC tumor annotator") as app:
        gr.Markdown(
            "## Gán nhãn KHỐI U — gross RCC (box→mask, SAM2.1@1024)\n"
            "**Khoanh RIÊNG khối u, chừa mô thận lành/vỏ.** Ảnh chưa gán có sẵn "
            "**auto-box (khung xanh dương)** — KIỂM lại: sát u thì **Box OK → tạo mask**, "
            "lệch thì tự vẽ Box (2 click) rồi bấm. Sau đó tinh chỉnh +/−điểm → "
            "**➕ Thêm u** nếu nhiều u/mảnh → **Accept&Lưu**. "
            "Đối chiếu *Excel mass* để mask không lớn hơn kích u nhiều.")
        idx = gr.State(0); committed_s = gr.State([])
        box_s = gr.State(None); corner_s = gr.State(None)
        pos_s = gr.State([]); neg_s = gr.State([])
        cur_s = gr.State(None); score_s = gr.State(None)
        with gr.Row():
            with gr.Column(scale=4):
                img = gr.Image(label="Ảnh (click để prompt)", type="numpy",
                               interactive=True, height=720)
            with gr.Column(scale=1):
                mode = gr.Radio(["Box (2 click)", "+điểm", "-điểm"],
                                value="Box (2 click)", label="Chế độ click")
                status_box = gr.Textbox(label="Trạng thái", lines=5, interactive=False)
                btn_auto = gr.Button("✅ Box OK → tạo mask", variant="secondary")
                btn_add = gr.Button("➕ Thêm u (ảnh nhiều mảnh)")
                btn_acc = gr.Button("Accept & Lưu → ảnh kế", variant="primary")
                btn_skip = gr.Button("Skip → ảnh kế")
                with gr.Row():
                    btn_clear = gr.Button("Clear prompt")
                    btn_clrall = gr.Button("Bỏ u đã thêm")
                with gr.Row():
                    btn_prev = gr.Button("◀ Prev")
                    btn_next = gr.Button("Next ▶")

        OUT_FULL = [idx, committed_s, box_s, corner_s, pos_s, neg_s, cur_s, img, status_box]
        img.select(click,
                   [idx, mode, committed_s, box_s, corner_s, pos_s, neg_s],
                   [committed_s, box_s, corner_s, pos_s, neg_s, cur_s, score_s, img, status_box])
        btn_auto.click(run_autobox, [idx, committed_s, box_s, pos_s, neg_s],
                       [committed_s, box_s, corner_s, pos_s, neg_s, cur_s, score_s, img, status_box])
        btn_add.click(add_obj, [idx, committed_s, cur_s],
                      [committed_s, box_s, corner_s, pos_s, neg_s, cur_s, img, status_box])
        btn_clear.click(clear_cur, [idx, committed_s],
                        [box_s, corner_s, pos_s, neg_s, cur_s, img, status_box])
        btn_clrall.click(clear_all, [idx],
                         [committed_s, box_s, corner_s, pos_s, neg_s, cur_s, img, status_box])
        btn_acc.click(accept, [idx, committed_s, cur_s, box_s, pos_s, neg_s, score_s], OUT_FULL)
        btn_skip.click(skip, [idx], OUT_FULL)
        btn_prev.click(lambda i: goto(i, i - 1), [idx], OUT_FULL)
        btn_next.click(lambda i: goto(i, i + 1), [idx], OUT_FULL)
        app.load(on_load, None, OUT_FULL)
    return app


if __name__ == "__main__":
    app = build_app()
    app.queue().launch(server_name="0.0.0.0", server_port=PORT, share=False)
