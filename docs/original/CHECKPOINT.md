# CHECKPOINT — chuẩn bị dữ liệu (tự sinh bởi run_all.sh)

## BEST CURRENT FULL-AUTO RUN — 2026-06-27

**Bản chạy tốt nhất cho đến hiện tại để dùng trên toàn bộ ca:** `specimen_strict` full-auto.

Đường dẫn kết quả:
- Summary: `results/full_auto_specimen_strict_20260626/summary.csv`
- Boxes: `results/full_auto_specimen_strict_20260626/boxes.csv`
- Masks: `results/full_auto_specimen_strict_20260626/masks/`
- Overlays: `results/full_auto_specimen_strict_20260626/overlays_thumb/`

Lệnh đã chạy:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/hvusynh2/conda_envs/medsam2_anno/bin/python -u predict_full_auto_all.py --out results/full_auto_specimen_strict_20260626 --overwrite --specimen_strict --min_box_spec_frac 0.50
```

Kết quả kiểm tra:
- Đã chạy đủ `1393/1393` ca.
- Có `1393` mask và `1393` overlay.
- `mask_outside_specimen = 0.0` cho toàn bộ ca, tức là đã chặn được lỗi tô đè ra background xanh/tím.
- `mask/specimen` median `0.3373`, mean `0.4677`.
- `n_box` median `4`, mean `4.43`, có `7` ca không detect được box.
- Có `399` ca `mask/specimen < 0.10`; vấn đề còn lại là detector/segmentation bỏ sót hoặc bắt quá ít vùng, không còn là lỗi tràn nền.

Ghi chú quan trọng: baseline cũ `detector_recall + SAM2.1 tiny` vẫn có Dice cao hơn một chút trên 50 handdraw (`median ~0.6659`) nhưng bị rủi ro tràn mask ra background. Với yêu cầu chạy thực tế toàn bộ data và tránh tô nền, `full_auto_specimen_strict_20260626` là bản tốt nhất hiện tại.

## CURRENT BEST DIRECTION — component/slice-aware strict inference

**Hướng đi hiện tại đúng nhất:** tách background trước, sau đó tách từng lát cắt/specimen component, detect box riêng trong từng lát, và clip mask SAM về đúng lát đó. Lý do: lỗi còn lại không phải max object bị set = 1, mà là ảnh nhiều lát khiến box/mask dồn vào một lát, bỏ sót lát khác hoặc lem trong nội bộ specimen.

Đã thêm code:
- `component_strict.py`
- `eval_component_strict.py`
- `predict_full_auto_all.py --component_strict`

Kiểm tra max object:
- Không có `max_instances` / `max_detections_per_image` / `max_det` bị set bằng `1`.
- `decode_detections(..., max_box=20)` và `decode_detections_specimen(..., max_box=20)`.
- Output full-data specimen-strict có `n_box` từ `0` đến `15`, tổng `6177` box. Vì vậy hiện tượng ảnh chỉ có 1 box là do detector/threshold/filter, không phải cấu hình giới hạn object.

Eval component-strict trên 50 handdraw + demo `SS21-38576`:
```text
strict050              median=0.6434 mean=0.5930 multi-u=0.5499 mask/spec=0.310
comp4_frac045_fb025    median=0.6408 mean=0.6032 multi-u=0.5871 mask/spec=0.273
comp2_frac050_fb035    median=0.6203 mean=0.5736 multi-u=0.5502 mask/spec=0.195
```

Kết luận:
- `specimen_strict` vẫn là **bản full-data ổn định tốt nhất đã chạy đủ 1393 ca**.
- `component_strict comp4` là **hướng thử nghiệm tốt nhất hiện tại cho ảnh nhiều lát/nhiều u**: mean tăng, multi-u tăng rõ, mask/spec giảm.
- `component_strict comp2` là bản conservative hơn nếu ưu tiên giảm lem hơn recall.

Output thử nghiệm:
- `results/component_strict_eval/component_strict_eval.csv`
- `results/component_strict_eval/component_strict_summary.json`
- `results/component_strict_eval/SS21-38576_strict_vs_component_contact.jpg`
- `results/component_strict_demo_SS21_38576/`
- `results/component_strict_demo_SS21_38576_comp2/`

Lệnh chạy full-data tiếp theo nếu muốn thử hướng đang đúng nhất:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/hvusynh2/conda_envs/medsam2_anno/bin/python -u predict_full_auto_all.py --out results/full_auto_component_strict_comp4_20260627 --overwrite --component_strict --component_max_box 4 --component_min_box_spec_frac 0.45 --component_fallback_thr 0.25
```

Lệnh ít lem hơn:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/hvusynh2/conda_envs/medsam2_anno/bin/python -u predict_full_auto_all.py --out results/full_auto_component_strict_comp2_20260627 --overwrite --component_strict --component_max_box 2 --component_min_box_spec_frac 0.50 --component_fallback_thr 0.35
```

_Sinh lúc: 2026-06-19 03:38:17_

## 1) Split train/val (theo bệnh nhân, 0 rò rỉ)
```
Tổng: 55 ảnh / 39 bệnh nhân
TRAIN: 43 ảnh / 29 bệnh nhân
VAL  : 12 ảnh / 10 bệnh nhân (~22% ảnh)
Không rò rỉ bệnh nhân ✔
-> labels/split.json , labels/split.csv
```
Chi tiết: `labels/split.json`, `labels/split.csv`

## 2) Lọc ảnh mặt-cắt vs mặt-ngoài (toàn 1392)
```
Ngưỡng cut_surface_score = 0.516  (phân vị 5% của 55 ảnh đã gán)
Nhóm đã gán: min 0.34 / med 0.82 / max 1.28  -> giữ 52/55
Toàn bộ: giữ 1128/1392 ảnh là MẶT-CẮT (81%), loại 264 (mặt-ngoài/nghi)
-> processed/cut_surface_filter.csv ; results/cut_filter_montage.jpg
```
Chi tiết: `processed/cut_surface_filter.csv` — QC ngưỡng: `results/cut_filter_montage.jpg`

## 3) Sanity-check diện tích khối u (cm², chuẩn hoá px/cm)
```
Tổng nhãn: 55 | dùng được (có px/cm & Excel cm²): 28
  bỏ vì thiếu px/cm: 27 | thiếu Excel cm²: 0

Spearman(tumor_cm2 ↔ Excel mass_cm2) = 0.088  (n=28)
  >0.5 = mask to/nhỏ ĐÚNG theo kích u thật -> tin được
tỉ lệ tumor_cm2/Excel_cm2: med 2.03 (min 0.05, max 119.23)
  ~0.5-1.0 hợp lý (mask 2D < diện tích từ caliper 3D).

[soi lại] 15 ảnh mask LỚN bất thường (>1.8x Excel) — có thể còn dính mô lành:
  SS21-39246^2021_06_25_08_31_34^^   tumor=268.3cm2 Excel=  2.2cm2 x119.2
  SS21-37297^2021_06_17_08_30_20^^   tumor= 82.8cm2 Excel=  1.4cm2 x57.5
  SS21-35913^2021_06_11_07_40_56^^   tumor=230.2cm2 Excel=  7.6cm2 x30.3
  SS21-35913^2021_06_10_10_46_24^^   tumor=183.3cm2 Excel=  7.6cm2 x24.1
  SS21-38576^2021_06_23_07_44_22^^   tumor= 72.9cm2 Excel=  7.6cm2 x9.6
  SS21-41214^2021_07_05_07_36_15^^   tumor= 37.5cm2 Excel=  5.9cm2 x6.3
  SS21-38576^2021_06_22_09_36_32^^   tumor= 33.2cm2 Excel=  7.6cm2 x4.4
  SS21-43481^2021_07_14_07_51_58^^   tumor=146.7cm2 Excel= 37.0cm2 x4.0
```

## 4) Hai việc BẢN CHẤT cần con người (chưa làm)
- **(a) Xác nhận thước:** liếc vài ảnh trong `processed/qc/` xem 1 vạch thước = 1cm không.
  Nếu không phải 1cm, chạy lại preprocess với `--tick_cm <đúng>` thì px/cm & cm² mới chuẩn.
- **(b) Quyết hướng model:** zero-shot SAM2.1 đã đủ, hay fine-tune trên 55 nhãn,
  hay gán thêm nhãn trước (box->mask cần người vẽ box).

## Đầu ra đã tạo
- `labels/split.json`, `labels/split.csv`
- `processed/metadata.csv` (px/cm, ruler_conf, cut_surface_score x1392)
- `processed/cut_surface_filter.csv`
- `results/cut_filter_montage.jpg`
