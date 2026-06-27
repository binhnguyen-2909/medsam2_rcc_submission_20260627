"""
APP VẼ TAY ĐỘC LẬP (KHÔNG SAM) — tạo test set khách quan, phá vòng GT-do-SAM.
Bút cọ tô vùng u trên canvas trắng (KHÔNG hiện mask SAM cũ → không thiên vị).
Lưu mask nhị phân full-res vào labels_handdraw/masks/{stem}.png, resume được.
Chạy: /home/hvusynh2/conda_envs/medsam2_anno/bin/python -u annotate_handdraw.py
  -> 0.0.0.0:18864  (VS Code forward cổng 18864)
KHÔNG import sam2 — đảm bảo độc lập tuyệt đối với mô hình.
"""
import os, sys, json, csv, numpy as np, cv2, gradio as gr
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT)
IMG_DIR = os.path.join(ROOT, "data/20241212")
EXCEL_CSV = os.path.join(ROOT, "processed/excel_parsed.csv")
OUT_DIR = os.path.join(ROOT, "labels_handdraw")
MASK_DIR = os.path.join(OUT_DIR, "masks"); OVL_DIR = os.path.join(OUT_DIR, "overlays")
DONE_JSON = os.path.join(OUT_DIR, "done.json"); SKIP_JSON = os.path.join(OUT_DIR, "skipped.json")
CSV_LOG = os.path.join(OUT_DIR, "handdraw_log.csv")
MAXDIM = 1500  # cỡ hiển thị để vẽ mượt; mask lưu lại full-res
os.makedirs(MASK_DIR, exist_ok=True); os.makedirs(OVL_DIR, exist_ok=True)

STEMS = json.load(open(os.path.join(OUT_DIR, "select.json")))["stems"]

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
    """Đọc ảnh, scale xuống MAXDIM để vẽ. Trả (rgb_disp, (origW,origH))."""
    bgr = cv2.imread(os.path.join(IMG_DIR, stem + ".jpg"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    sc = MAXDIM / max(H, W)
    if sc < 1.0:
        rgb = cv2.resize(rgb, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
    return rgb, (W, H)

def info_text(i):
    stem = STEMS[i]; done = set(load_json(DONE_JSON, [])); skip = set(load_json(SKIP_JSON, []))
    st = "✅ đã lưu" if stem in done else ("⏭️ skip" if stem in skip else "⬜ chưa")
    row = EXCEL.get(canon_of(stem), {})
    mass = row.get("mass_dims") or row.get("mass_snippet") or "—"
    nm = row.get("n_mass", "?")
    nd = len(done)
    return (f"### {i+1}/{len(STEMS)} — {st}\n"
            f"`{stem}`\n\n"
            f"**Kích u (Excel, sanity-check — KHÔNG phải vị trí):** {mass}  |  n_mass≈{nm}\n\n"
            f"Đã lưu: **{nd}/{len(STEMS)}**\n\n"
            f"➡️ Tô **TẤT CẢ** vùng u (nhiều mảnh thì tô hết). Vẽ tự do, KHÔNG nhìn gợi ý máy.")

def editor_value(stem):
    rgb, _ = disp_image(stem)
    return {"background": rgb, "layers": [], "composite": rgb}

def extract_mask(edit, origWH):
    """Lấy mask từ nét vẽ (alpha của layers), resize về full-res."""
    W, H = origWH
    m = None
    if isinstance(edit, dict):
        layers = edit.get("layers") or []
        for ly in layers:
            ly = np.asarray(ly)
            if ly.ndim == 3 and ly.shape[2] == 4:
                a = ly[..., 3] > 10
                m = a if m is None else (m | a)
        if m is None:  # fallback: composite khác background
            bg = np.asarray(edit.get("background")); comp = np.asarray(edit.get("composite"))
            if bg is not None and comp is not None and bg.shape == comp.shape:
                m = np.abs(comp.astype(int) - bg.astype(int)).sum(-1) > 25
    if m is None or m.sum() == 0: return None
    mu = (m.astype(np.uint8)) * 255
    if (mu.shape[1], mu.shape[0]) != (W, H):
        mu = cv2.resize(mu, (W, H), interpolation=cv2.INTER_NEAREST)
    return (mu > 127).astype(np.uint8) * 255

# ---------------- handlers ----------------
def on_load(i):
    i = int(i) % len(STEMS); stem = STEMS[i]
    return editor_value(stem), info_text(i), i

def on_save(edit, i):
    i = int(i); stem = STEMS[i]; _, WH = disp_image(stem)
    mask = extract_mask(edit, WH)
    if mask is None:
        return info_text(i) + "\n\n⚠️ **Chưa vẽ gì** — tô vùng u rồi Lưu, hoặc bấm Skip.", i
    cv2.imwrite(os.path.join(MASK_DIR, stem + ".png"), mask)
    # overlay QC
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
    ni = (i + 1) % len(STEMS)
    return editor_value(STEMS[ni]), info_text(ni), ni

def on_skip(i):
    i = int(i); stem = STEMS[i]; skip = load_json(SKIP_JSON, [])
    if stem not in skip: skip.append(stem); save_json(SKIP_JSON, skip)
    ni = (i + 1) % len(STEMS)
    return editor_value(STEMS[ni]), info_text(ni), ni

def on_nav(i, d):
    ni = (int(i) + d) % len(STEMS)
    return editor_value(STEMS[ni]), info_text(ni), ni

with gr.Blocks(title="Vẽ tay test set (không SAM)") as demo:
    gr.Markdown("# ✍️ Vẽ tay mask u — TEST SET ĐỘC LẬP (không dùng SAM)")
    idx = gr.State(first_todo())
    with gr.Row():
        with gr.Column(scale=3):
            editor = gr.ImageEditor(label="Tô vùng u (bút cọ đỏ). Tẩy để sửa.",
                                    type="numpy", image_mode="RGB",
                                    brush=gr.Brush(default_size=22, colors=["#FF0000"], color_mode="fixed"),
                                    layers=False, transforms=(), sources=(), height=720)
        with gr.Column(scale=1):
            info = gr.Markdown()
            save_btn = gr.Button("💾 Lưu & Tiếp →", variant="primary")
            skip_btn = gr.Button("⏭️ Skip (không có u rõ / ảnh xấu)")
            with gr.Row():
                prev_btn = gr.Button("← Trước"); next_btn = gr.Button("Sau →")
            gr.Markdown("*Tự lưu vào `labels_handdraw/`. Resume: mở lại app tự nhảy ảnh chưa làm.*")

    demo.load(on_load, [idx], [editor, info, idx])
    save_btn.click(on_save, [editor, idx], [editor, info, idx])
    skip_btn.click(on_skip, [idx], [editor, info, idx])
    prev_btn.click(lambda i: on_nav(i, -1), [idx], [editor, info, idx])
    next_btn.click(lambda i: on_nav(i, +1), [idx], [editor, info, idx])

if __name__ == "__main__":
    print(f"{len(STEMS)} ảnh | đã lưu {len(load_json(DONE_JSON, []))} | bắt đầu ở #{first_todo()+1}", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=18864, show_error=True)
