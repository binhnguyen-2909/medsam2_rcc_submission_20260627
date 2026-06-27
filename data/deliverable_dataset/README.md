# Bộ mask khối u RCC — dataset cuối (1020 ảnh đã gán)

Mask khối u trên ảnh đại thể mặt cắt thận RCC, gán nửa-tự-động qua
[`annotate.py`](../annotate.py) (box→SAM2.1@1024 + tinh chỉnh điểm + union nhiều mảnh).

## Nội dung
- `masks/<stem>.png` — mask nhị phân full-res (0=nền, 255=u), cùng kích thước ảnh gốc.
- `manifest.csv` — 1 dòng/ảnh:

| cột | nghĩa |
|---|---|
| `stem` | mã ảnh (= tên file, không đuôi) |
| `patient_id` | mã bệnh nhân (phần trước `^`) |
| `image` | đường dẫn ảnh gốc (`data/20241212/<stem>.jpg`) |
| `mask` | đường dẫn mask trong dataset này |
| `W`,`H` | kích thước ảnh |
| `n_objects` | số mảnh/khối u trong ảnh |
| `union_area_px` | tổng diện tích u (px) |
| `n_boxes_saved`,`full_box` | số box lưu được; `1`=đủ box mọi instance |
| `boxes` | box đã lưu, dạng `x0,y0,x1,y1;x0,y0,...` (có thể thiếu ở ảnh nhiều-object) |
| `mass_dims_cm`,`mass_area_cm2` | kích u từ Excel (sanity-check, KHÔNG phải nhãn) |

## Lưu ý
- Đây là **bộ mask hoàn chỉnh** (đã tinh chỉnh tay, đủ mọi mảnh u) — khác với mask
  sinh bằng `predict_box2mask.py` thuần box (không tinh chỉnh điểm).
- ⚠️ Mask tạo có SAM2.1 hỗ trợ → dùng làm train/tham chiếu tốt, nhưng để đánh giá
  model KHÁCH QUAN cần test set vẽ tay độc lập (chưa có).
- `box` chỉ đầy đủ ở ảnh đơn-object (550 ảnh); ảnh nhiều-object thường thiếu box các
  mảnh thêm sau (giới hạn cách lưu của annotate.py) — mask vẫn đầy đủ, chỉ prompt thiếu.
