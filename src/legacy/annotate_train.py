"""
APP SỬA/THÊM MASK TRAIN (nâng chất supervision detector) — KHÔNG import sam2.
Khác app test (annotate_handdraw): PREFILL mask SAM cũ làm lớp đỏ chỉnh sửa được, nên
chỉ cần THÊM mảnh u bị sót + CẮT phần thừa (nhanh). Lưu mask THẬT vào labels_truth/masks.
Mục tiêu: vá gốc rễ "detector học từ nhãn SAM đếm-sót-mảnh" -> dạy lại bằng mask đầy đủ.
Queue = ảnh TRAIN (đã loại eval200/test12/50-ảnh-vẽ-tay). Resume được.
Chạy: /home/hvusynh2/conda_envs/medsam2_anno/bin/python -u annotate_train.py
  -> 0.0.0.0:18865  (VS Code forward cổng 18865)
"""
import os, sys, json, csv, numpy as np, cv2, gradio as gr
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT)
IMG_DIR = os.path.join(ROOT, "data/20241212")
EXCEL_CSV = os.path.join(ROOT, "processed/excel_parsed.csv")
SAM_MASK = os.path.join(ROOT, "labels/masks")          # mask SAM cũ để prefill
OUT_DIR = os.path.join(ROOT, "labels_truth")
MASK_DIR = os.path.join(OUT_DIR, "masks"); OVL_DIR = os.path.join(OUT_DIR, "overlays")
DONE_JSON = os.path.join(OUT_DIR, "done.json"); SKIP_JSON = os.path.join(OUT_DIR, "skipped.json")
CSV_LOG = os.path.join(OUT_DIR, "truth_log.csv")
MAXDIM = 1500
os.makedirs(MASK_DIR, exist_ok=True); os.makedirs(OVL_DIR, exist_ok=True)

STEMS = json.load(open(os.path.join(OUT_DIR, "queue.json")))["stems"]

def patient_of(s): return s.split("^")[0]
def canon_of(s): return patient_of(s).replace("-", "")
def load_excel():
    m = {}
    if os.path.isfile(EXCEL_CSV):
        for r in csv.DictReader(open(EXCEL_CSV)): m[r["canon"]] = r
    return m
EXCEL = load_excel()
def load_json(p, d): return json.load(open(p)) if os.path.isfile(p) else d
def save_json(p, o): json.dump(o, open(p, "w"), indent=1, ensure_ascii=False)

def first_todo():
    done = set(load_json(DONE_JSON, [])); skip = set(load_json(SKIP_JSON, []))
    for i, s in enumerate(STEMS):
        if s not in done and s not in skip: return i
    return 0

def disp_image(stem):
    bgr = cv2.imread(os.path.join(IMG_DIR, stem + ".jpg"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); H, W = rgb.shape[:2]
    sc = MAXDIM / max(H, W)
    if sc < 1.0: rgb = cv2.resize(rgb, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
    return rgb, (W, H)

def prefill_layer(stem, disp_hw):
    """mask đã lưu (labels_truth) nếu có, else mask SAM cũ -> lớp RGBA đỏ cùng cỡ hiển thị."""
    dh, dw = disp_hw
    src = os.path.join(MASK_DIR, stem + ".png")
    if not os.path.isfile(src): src = os.path.join(SAM_MASK, stem + ".png")
    if not os.path.isfile(src): return None
    m = cv2.imread(src, 0)
    if m is None: return None
    m = cv2.resize(m, (dw, dh), interpolation=cv2.INTER_NEAREST) > 127
    rgba = np.zeros((dh, dw, 4), np.uint8)
    rgba[m] = (255, 0, 0, 180)
    return rgba

def info_text(i):
    stem = STEMS[i]; done = set(load_json(DONE_JSON, [])); skip = set(load_json(SKIP_JSON, []))
    st = "✅ đã lưu" if stem in done else ("⏭️ skip" if stem in skip else "⬜ chưa")
    row = EXCEL.get(canon_of(stem), {})
    mass = row.get("mass_dims") or row.get("mass_snippet") or "—"
    nm = row.get("n_mass", "?"); nd = len(done)
    pre = "🔴 prefill = mask SAM cũ" if not os.path.isfile(os.path.join(MASK_DIR, stem + ".png")) else "🔴 mask đã lưu"
    return (f"### {i+1}/{len(STEMS)} — {st}\n`{stem}`\n\n"
            f"**Kích u (Excel sanity-check):** {mass}  |  n_mass≈{nm}\n\n"
            f"Đã lưu: **{nd}** (mục tiêu ~100–200)\n\n{pre}\n\n"
            f"➡️ **THÊM mảnh u SAM bỏ sót** + **TẨY phần tô lan ra mô lành/cả tạng**. "
            f"Tô hết MỌI mảnh u. Bút đỏ thêm, tẩy để bớt.")

def editor_value(stem):
    rgb, _ = disp_image(stem)
    ly = prefill_layer(stem, rgb.shape[:2])
    layers = [ly] if ly is not None else []
    comp = rgb.copy()
    if ly is not None:
        a = ly[..., 3:4].astype(np.float32) / 255.0
        comp = (rgb * (1 - a) + ly[..., :3] * a).astype(np.uint8)
    return {"background": rgb, "layers": layers, "composite": comp}

def extract_mask(edit, origWH):
    W, H = origWH; m = None
    if isinstance(edit, dict):
        for ly in (edit.get("layers") or []):
            ly = np.asarray(ly)
            if ly.ndim == 3 and ly.shape[2] == 4:
                a = ly[..., 3] > 10; m = a if m is None else (m | a)
        if m is None:
            bg = np.asarray(edit.get("background")); comp = np.asarray(edit.get("composite"))
            if bg is not None and comp is not None and bg.shape == comp.shape:
                m = np.abs(comp.astype(int) - bg.astype(int)).sum(-1) > 25
    if m is None or m.sum() == 0: return None
    mu = (m.astype(np.uint8)) * 255
    if (mu.shape[1], mu.shape[0]) != (W, H):
        mu = cv2.resize(mu, (W, H), interpolation=cv2.INTER_NEAREST)
    return (mu > 127).astype(np.uint8) * 255

def on_load(i):
    i = int(i) % len(STEMS); stem = STEMS[i]
    return editor_value(stem), info_text(i), i

def on_save(edit, i):
    i = int(i); stem = STEMS[i]; _, WH = disp_image(stem)
    mask = extract_mask(edit, WH)
    if mask is None:
        return editor_value(stem), info_text(i) + "\n\n⚠️ **Mask rỗng** — tô vùng u rồi Lưu, hoặc Skip.", i
    cv2.imwrite(os.path.join(MASK_DIR, stem + ".png"), mask)
    bgr = cv2.imread(os.path.join(IMG_DIR, stem + ".jpg")); ov = bgr.copy()
    mb = mask > 127; ov[mb] = (0.45 * ov[mb] + 0.55 * np.array([40, 40, 255])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (0, 255, 255), 3)
    cv2.imwrite(os.path.join(OVL_DIR, stem + ".png"), ov)
    done = load_json(DONE_JSON, [])
    if stem not in done: done.append(stem); save_json(DONE_JSON, done)
    skip = load_json(SKIP_JSON, [])
    if stem in skip: skip.remove(stem); save_json(SKIP_JSON, skip)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    area = int((mask > 127).sum()); ncc = int((stats[1:, cv2.CC_STAT_AREA] >= 0.002 * mask.size).sum())
    hdr = not os.path.isfile(CSV_LOG)
    with open(CSV_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if hdr: w.writerow(["stem", "patient", "area_px", "n_comp", "W", "H"])
        w.writerow([stem, patient_of(stem), area, ncc, WH[0], WH[1]])
    ni = next_todo_idx(i, +1)            # nhảy thẳng tới ảnh CHƯA đánh tiếp theo
    return editor_value(STEMS[ni]), info_text(ni), ni

def on_skip(i):
    i = int(i); stem = STEMS[i]; skip = load_json(SKIP_JSON, [])
    if stem not in skip: skip.append(stem); save_json(SKIP_JSON, skip)
    ni = next_todo_idx(i, +1)            # nhảy thẳng tới ảnh CHƯA đánh tiếp theo
    return editor_value(STEMS[ni]), info_text(ni), ni

def on_nav(i, d):
    ni = (int(i) + d) % len(STEMS)
    return editor_value(STEMS[ni]), info_text(ni), ni

def next_todo_idx(i, d=+1):
    """Nhảy tới ảnh CHƯA đánh (bỏ qua done+skip) theo hướng d. Không có thì giữ nguyên."""
    done = set(load_json(DONE_JSON, [])); skip = set(load_json(SKIP_JSON, []))
    n = len(STEMS)
    for k in range(1, n + 1):
        j = (int(i) + d * k) % n
        if STEMS[j] not in done and STEMS[j] not in skip:
            return j
    return int(i)

def on_jump_todo(i, d):
    ni = next_todo_idx(i, d)
    return editor_value(STEMS[ni]), info_text(ni), ni

with gr.Blocks(title="Sửa/thêm mask TRAIN") as demo:
    gr.Markdown("# 🛠️ Sửa & thêm mask TRAIN (mask SAM prefill đỏ — thêm mảnh sót, cắt phần thừa)")
    idx = gr.State(first_todo())
    with gr.Row():
        with gr.Column(scale=3):
            editor = gr.ImageEditor(label="Đỏ = mask hiện có. Bút thêm mảnh sót, tẩy phần lan ra mô lành.",
                                    type="numpy", image_mode="RGB",
                                    brush=gr.Brush(default_size=22, colors=["#FF0000"], color_mode="fixed"),
                                    layers=False, transforms=(), sources=(), height=720)
        with gr.Column(scale=1):
            info = gr.Markdown()
            save_btn = gr.Button("💾 Lưu & Tiếp →", variant="primary")
            skip_btn = gr.Button("⏭️ Skip (không có u / ảnh xấu)")
            with gr.Row():
                prev_btn = gr.Button("← Trước"); next_btn = gr.Button("Sau →")
            with gr.Row():
                todo_prev_btn = gr.Button("⏭️ ⟵ Chưa đánh"); todo_next_btn = gr.Button("Chưa đánh ⟶ ⏭️", variant="secondary")
            gr.Markdown("*Lưu/Skip + nút \"Chưa đánh\" nhảy thẳng tới ảnh chưa làm (bỏ qua done+skip). ←/Sau→ đi tuần tự.*")
    demo.load(on_load, [idx], [editor, info, idx])
    save_btn.click(on_save, [editor, idx], [editor, info, idx])
    skip_btn.click(on_skip, [idx], [editor, info, idx])
    prev_btn.click(lambda i: on_nav(i, -1), [idx], [editor, info, idx])
    next_btn.click(lambda i: on_nav(i, +1), [idx], [editor, info, idx])
    todo_prev_btn.click(lambda i: on_jump_todo(i, -1), [idx], [editor, info, idx])
    todo_next_btn.click(lambda i: on_jump_todo(i, +1), [idx], [editor, info, idx])

if __name__ == "__main__":
    print(f"{len(STEMS)} ảnh train | đã lưu {len(load_json(DONE_JSON, []))} | bắt đầu #{first_todo()+1}", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=18865, show_error=True)
