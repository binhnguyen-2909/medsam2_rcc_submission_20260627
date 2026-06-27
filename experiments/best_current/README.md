# Best Current Run

Bảng này chỉ trích ra **cách chạy tốt nhất cho đến hiện tại** đã được ghi trong note gốc (`docs/original/CHECKPOINT.md`, `docs/original/TONG_KET_HANH_TRINH.md`). Các hướng đã thử nhưng bị loại nằm ở `experiments/rejected/README.md`.

| Mục | Nội dung |
|---|---|
| Tên run | `full_auto_specimen_strict_20260626` |
| Loại run | Full-auto stable run trên toàn bộ dataset |
| Lệnh thực thi | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/hvusynh2/conda_envs/medsam2_anno/bin/python -u predict_full_auto_all.py --out results/full_auto_specimen_strict_20260626 --overwrite --specimen_strict --min_box_spec_frac 0.50` |
| Script tương ứng trong package này | `src/legacy/predict_full_auto_all.py` |
| Config/thông số chính | `--specimen_strict`; `--min_box_spec_frac 0.50`; detector/checkpoint theo script gốc; output mask/overlay cho toàn bộ ảnh |
| Cơ sở lý thuyết | Full-auto cũ có Dice handdraw tốt hơn nhẹ nhưng có rủi ro tô mask ra background xanh/tím. `specimen_strict` dùng specimen mask để gate/clip box và mask, ưu tiên độ ổn định khi chạy toàn bộ dataset. |
| Kết quả | Đã chạy đủ `1393/1393` ca; sinh `1393` mask và `1393` overlay; `mask_outside_specimen = 0.0`; `mask/specimen` median `0.3373`, mean `0.4677`; `n_box` median `4`, mean `4.43`; `7` ca không detect box; `399` ca `mask/specimen < 0.10`. |
| Output trong package này | `results/original/full_auto_specimen_strict_20260626/summary.csv`; `results/original/full_auto_specimen_strict_20260626/boxes.csv`; `results/original/full_auto_specimen_strict_20260626/masks/`; `results/original/full_auto_specimen_strict_20260626/overlays_thumb/` |
| Phân tích | Đây là bản **full-data ổn định tốt nhất đã chạy xong** vì loại được lỗi nghiêm trọng nhất là tô ra ngoài specimen/background. Baseline detector+SAM có median Dice cao hơn nhẹ trên 50 handdraw nhưng không ổn định bằng khi chạy toàn bộ dataset. |
| Lý do không chọn hướng khác làm best-current | `component_strict comp4` cải thiện mean và multi-u trên 50 handdraw nhưng median chưa vượt `specimen_strict`, và chưa phải bản stable full-data đã chốt. Các hướng model/loss/localizer khác đã bị bác trong `experiments/rejected/README.md`. |
| Logic bước tiếp theo | Nếu tiếp tục tối ưu full-auto, chạy `component_strict comp4` trên toàn bộ data để kiểm chứng ảnh nhiều lát: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/hvusynh2/conda_envs/medsam2_anno/bin/python -u predict_full_auto_all.py --out results/full_auto_component_strict_comp4_20260627 --overwrite --component_strict --component_max_box 4 --component_min_box_spec_frac 0.45 --component_fallback_thr 0.25`. |

## Best Semi-auto Deliverable

Nếu mục tiêu là **kết quả mask tốt nhất khi có người vẽ/duyệt box**, dùng deliverable hiện tại:

| Mục | Nội dung |
|---|---|
| Tên run/model | `seg_crop_segR_lab` |
| Loại run | Semi-auto box-to-mask deliverable |
| Lệnh thực thi mẫu | `python src/inference/predict_seg_crop.py --config configs/model/seg_crop_lab.yaml --image data/raw/images/CASE.jpg --box "x0,y0,x1,y1" --out experiments/runs/RUN_ID/predictions/CASE.png --overlay experiments/runs/RUN_ID/predictions/CASE_overlay.jpg` |
| Config/thông số chính | `configs/model/seg_crop_lab.yaml`; checkpoint `checkpoints/seg_crop_segR_lab.pt`; SegResNet; RGB+LAB; crop pad `0.15`; threshold `0.5`; input crop `512x512` |
| Cơ sở lý thuyết | Khi box đúng, lỗi chính của SAM là spill sang mô lành. Segmenter crop chuyên dụng học mask u trong vùng crop và giảm spill tốt hơn SAM box-prompt. |
| Kết quả | Handdraw50 ceiling median Dice `0.8834`, mean `0.8682`; tốt hơn SAM box-to-mask cũ khoảng `0.857`. |
| Output/config trong package này | `src/inference/predict_seg_crop.py`; `configs/model/seg_crop_lab.yaml`; `checkpoints/seg_crop_segR_lab.pt`; `results/tables/seg_crop_segR_lab.json` |
| Phân tích | Đây là **deliverable chất lượng mask tốt nhất hiện tại** nếu quy trình cho phép người vẽ hoặc duyệt box. Không phải full-auto, nhưng đáng tin cậy hơn full-auto vì localize vẫn là nút thắt. |
