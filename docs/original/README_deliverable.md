# Deliverable — Box→Mask khối u RCC (ảnh đại thể)

Sinh **mask nhị phân khối u** từ **bounding-box do người vẽ** trên ảnh đại thể (gross
pathology) mặt cắt thận RCC. Dùng **SAM2.1 vanilla tiny @1024, ZERO-SHOT** (không
fine-tune). Script: [`predict_box2mask.py`](predict_box2mask.py).

## Vì sao zero-shot, không fine-tune?
Đã thử fine-tune ở 47 / 170 / **1008** ảnh train (test 12 ảnh đóng băng) — không mốc
nào vượt zero-shot (đều Dice 0.958; FT overfit: train_loss giảm nhưng test Dice tụt
0.93-0.94). Thêm dữ liệu 6× cũng không đổi → **giao zero-shot**.

## Yêu cầu
- Python env: `/home/hvusynh2/conda_envs/medsam2_anno/bin/python` (hoặc `medsam2`, có hydra + sam2)
- Checkpoint: `checkpoints/sam2.1_hiera_tiny.pt` (đã có)
- GPU khuyến nghị (chạy CPU được nhưng chậm)
- Cấu hình qua env (mặc định đã đúng):
  `SAM2_CONFIG=configs/sam2.1_hiera_t512  SAM2_CKPT=checkpoints/sam2.1_hiera_tiny.pt  SAM2_RES=1024`

## Dùng

**1 ảnh, 1 box** (toạ độ pixel trên ảnh GỐC: `x0,y0`=trên-trái, `x1,y1`=dưới-phải):
```bash
python predict_box2mask.py --image a.jpg --box "x0,y0,x1,y1" --out a_mask.png
```

**1 ảnh, nhiều box** (ảnh có nhiều mảnh/nhiều u → union):
```bash
python predict_box2mask.py --image a.jpg \
    --box "120,80,400,350" --box "600,500,820,700" \
    --out a_mask.png --overlay a_overlay.jpg
```

**Batch từ CSV** (cột `image,x0,y0,x1,y1`; nhiều dòng cùng `image` → union):
```bash
python predict_box2mask.py --csv boxes.csv --out_dir out_masks --overlay_dir out_ov
```
Ví dụ `boxes.csv`:
```csv
image,x0,y0,x1,y1
case001.jpg,436,666,704,902
case002.jpg,417,723,732,951
case002.jpg,900,400,1100,650
```
`image` nhận đường dẫn tuyệt đối/tương đối, hoặc chỉ stem (tự tìm trong `data/20241212/`).

## Output
- Mask PNG nhị phân **full-res** cùng kích thước ảnh gốc: `0`=nền, `255`=khối u.
- Overlay JPG (tuỳ chọn): mask xanh + viền vàng + box xanh dương để QC bằng mắt.

## Chất lượng & lưu ý
- Dice vs nhãn nội bộ ~**0.96** (kiểm trên test đóng băng).
- ⚠️ Nhãn "ground-truth" nội bộ do SAM2.1 hỗ trợ tạo → số Dice này **thiên vị cao**;
  muốn đánh giá khách quan cần test set **vẽ tay độc lập** (chưa làm).
- Box phải bao **RIÊNG khối u** (chừa mô thận lành/vỏ). Box quá rộng → SAM dễ nuốt cả
  mảng mô lành. Auto-box gợi ý (trong tool gán nhãn) chỉ để prefill, **người phải siết lại**.
- Gán nhãn tương tác (vẽ box + tinh chỉnh điểm): dùng [`annotate.py`](annotate.py).
