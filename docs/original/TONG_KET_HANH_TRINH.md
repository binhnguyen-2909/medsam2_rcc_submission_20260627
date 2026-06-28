# Tổng kết hành trình dự án — Segment vùng u RCC trên ảnh đại thể

> Mục tiêu: phát hiện/khoanh (segment) **khối u** trên **ảnh đại thể (gross pathology)** mặt cắt thận RCC
> — ảnh chụp máy ảnh bệnh phẩm đã mổ (2736×1824, nền tím), KHÔNG phải X-quang/CT/siêu âm.
>
> Tài liệu này ghi lại **từng bước đã làm, vì sao làm, chỗ nào sai, sửa thế nào, tại sao, và so sánh trước/sau**.
> Đọc kèm: `CHECKPOINT.md`, `README_deliverable.md`, các file trong `results/`.

---

## 0. Bức tranh kết quả cuối (đọc trước để có khung quy chiếu)

Mọi con số dưới đây đo trên **50 ảnh vẽ tay độc lập** (ground-truth KHÔNG qua SAM — xem Bước 9):

| Cấu hình | Dice median | Ý nghĩa |
|---|---|---|
| Nhãn SAM cũ vs sự thật | **0.554** | Chất lượng nhãn tự sinh trước khi có test thật |
| Full-auto tốt nhất (detector → box → SAM) | **0.666** | Máy tự làm hoàn toàn — **TRẦN thực tế** |
| Segmenter chuyên dụng (seg_sam, encoder lớn) | 0.637 | Hướng thay box→SAM, vẫn thua 0.666 |
| **Trần box→mask (box đúng → SAM)** | **0.857** | Nếu box được vẽ/duyệt đúng |

**Hai kết luận xương sống:**
1. **Full-auto bị chặn ở ~0.666** — nút thắt là *định vị u* (localize) + *SAM tô lố ra mô lành* (spill), KHÔNG phải kích thước mô hình.
2. **Deliverable thực tế = SEMI-AUTO**: người duyệt/sửa box (vài giây) → SAM cho mask **0.857**. Đây là sản phẩm bàn giao.

---

## 1. Hiểu đúng dữ liệu & sửa sơ đồ bài toán SAI ban đầu

**Đã làm:** Xác minh bản chất 1392 ảnh + file Excel `RCC 20241212.xlsx`.

**Định hướng:** Trước khi code, phải biết dữ liệu là gì và nhãn ở đâu.

**SAI ban đầu — sơ đồ "kích thước cm → sinh bbox":**
- Ý tưởng gốc: lấy số cm trong Excel (vd "mass 6.0×5.5cm") để sinh hộp bao quanh u.
- **Tại sao sai:** cm chỉ cho *kích thước* u, KHÔNG cho *vị trí* u trong ảnh. Không thể đặt hộp nếu không biết u nằm đâu.

**Khắc phục & tại sao:**
- Reframe: cm chỉ dùng để **sanity-check diện tích mask** (so với diện tích vùng u, KHÔNG so với cả bệnh phẩm).
- Vị trí u phải đến từ **thị giác** (người click / mô hình học), không từ Excel.
- → Bài toán đúng = **box → mask** (cho hộp, sinh mask), KHÔNG phải "cm → mask".

**Trước/sau:** Trước = pipeline bế tắc (không có vị trí). Sau = hướng box→mask khả thi, là nền cho mọi thứ về sau.

---

## 2. Tiền xử lý ảnh (`preprocess.py`)

**Đã làm:** Tách nền tím (HSV) + connected-component → mask bệnh phẩm (tốt 12/12). Dò cây thước → px/cm. Xuất `processed/`.

**Định hướng:** Cô lập bệnh phẩm khỏi nền, và thử quy đổi pixel→cm để kiểm diện tích.

**SAI — px/cm từ dò thước không đáng tin:**
- Autocorrelation bắt nhầm bước vạch mm vs cm → px/cm lệch tới 5×; chỉ 28/55 ảnh có giá trị, Spearman(diện tích↔cm) chỉ 0.088.

**Khắc phục & tại sao:** **Bỏ px/cm** làm công cụ QC; tin QC bằng mắt. Vì sai số quá lớn và **không ảnh hưởng deliverable box→mask** (không cần cm để khoanh u).

**Trước/sau:** Trước = định kiểm tra tự động bằng cm (loạn). Sau = QC mắt thường, tập trung nguồn lực vào nhãn.

---

## 3. Chọn backbone — SAM2.1 vanilla vs MedSAM2 (`benchmark_backbones.py`)

**Đã làm:** Benchmark box-prompt ở 512px và 1024px.

**Định hướng:** Chọn mô hình nền tốt nhất cho box→mask trên domain ảnh màu.

**Kết quả:** **SAM2.1 vanilla thắng dứt khoát** (self-score 0.92–0.97 @1024) vs MedSAM2 (0.48–0.77 — lệch domain vì MedSAM2 huấn luyện cho ảnh y khoa xám, không hợp ảnh màu). Chốt **SAM2.1 vanilla @1024**.

---

## 4. Công cụ gán nhãn (`annotate.py`) & chuẩn hoá đối tượng khoanh

**Đã làm:** Viết app Gradio gán nhãn box→mask ảnh đơn (app gốc chỉ làm video). Resume được, đa-khối-u, re-review.

**SAI — vòng nhãn đầu: "SAM nuốt cả lát bệnh phẩm":**
- sam_score cao (median 0.93) **che giấu lỗi**: khi mặt cắt đồng màu, SAM khoanh cả lát thận thay vì riêng u (vd ca score 0.96 nhưng mask = cả lát 8cm trong khi u chỉ 3×2.8cm).

**Khắc phục & tại sao:**
- **Chốt định nghĩa đối tượng: CHỈ khối u** (chừa mô lành/vỏ); ảnh nhiều mảnh → gán TẤT CẢ mảnh có u (union).
- `flag_relabel.py` gắn cờ `fill_ratio>0.75` = nghi cả-lát để rà lại.
- Vì score cao ≠ đúng; phải định nghĩa rõ ràng cái cần khoanh.

**Trước/sau:** Trước = nhãn lẫn cả lát (sai bản chất). Sau = nhãn tumor-only nhất quán, 55 ảnh chốt đầu tiên.

---

## 5. Chuẩn bị dữ liệu sạch (`split_dataset.py`, `filter_cut_surface.py`, `area_sanity.py`)

**Đã làm:** Chia 43 train / 12 val theo **bệnh nhân** (0 rò rỉ, deterministic). Lọc mặt-cắt (giữ 1128/1392). `area_sanity` xác nhận lại px/cm không tin được (xem Bước 2).

**Định hướng:** Tách train/test **theo bệnh nhân** để tránh rò rỉ (2 ảnh cùng bệnh nhân không được nằm 2 bên).

---

## 6. Fine-tune SAM → thua zero-shot (`finetune_sam2.py`)

**Đã làm:** Đóng băng image encoder, train prompt_encoder + mask_decoder (4.22M params), loss Dice+BCE.

**Định hướng:** Giả định fine-tune trên data RCC sẽ tốt hơn zero-shot.

**SAI — giả định bị bác:**
- Zero-shot test Dice = 0.958 (epoch 0). Fine-tune 60 epoch **KHÔNG epoch nào vượt** (tụt còn 0.92–0.94 dù train_loss giảm mạnh = **overfit kinh điển**).
- Xác nhận quy mô lớn (N=200): zero-shot 0.9645 vs FT 0.9502, **FT thua 2:1** (68/132).

**Khắc phục & tại sao:** **Chốt zero-shot**, bỏ fine-tune. Vì data ít (43–1008) + nhãn do chính SAM hỗ trợ tạo → fine-tune chỉ học lại chính nó và overfit.

**Trước/sau:** Trước = kỳ vọng FT cải thiện. Sau = zero-shot 0.958 ≥ FT 0.950, đơn giản hơn và tốt hơn.

---

## 7. Backbone lớn (large) → bằng tiny → giữ tiny

**Đã làm:** Tải SAM2.1 **large** (857MB), so zero-shot tiny vs large trên 12 test.

**Định hướng:** Giả định mô hình lớn hơn → mask tốt hơn.

**SAI — giả định bị bác (lần 1):**
- tiny 0.958 vs large 0.946. Viền tiny/large/GT **trùng khít** → large cho mask y hệt nhưng ngốn GPU 6×.

**Khắc phục & tại sao:** **Giữ tiny.** Vì với box-prompt, hộp đã định vị sẵn → encoder lớn không thêm thông tin. Đòn bẩy thật là **thêm nhãn**, không phải model to.

---

## 8. Auto-box (cell-classifier) → chỉ dùng PREFILL

**Đã làm:** `train_cellbox.py` — lưới ô trên bệnh phẩm, phân loại ô-có-u → đề xuất box tự động (`propose_box.py`).

**Định hướng:** Để máy tự gợi ý box, đỡ công người định vị.

**SAI — "auto-box bấm thẳng SAM" không dùng được:**
- Dice median chỉ 0.330 (vs human-box 0.965); 5/12 "nổ" (mask >2× u). Box lỏng → SAM nuốt cả bệnh phẩm.

**Khắc phục & tại sao:** **Auto-box = PREFILL-ONLY** — hiện box gợi ý xanh dương, người DUYỆT (sát u thì OK, lệch thì vẽ lại). Vì localize tự động chưa đủ tin để bỏ người ra khỏi vòng.

**Trước/sau:** Trước = mơ one-click full-auto (0.33, hỏng). Sau = bán-tự-động có người duyệt (giữ chất lượng), auto-box chỉ giảm công.

---

## 9. ⭐ Bước then chốt: Test set VẼ TAY độc lập — phát hiện mọi Dice 0.95 là ẢO

**Đã làm:** User vẽ tay **50 mask** bằng bút cọ trên canvas trắng (`annotate_handdraw.py`, KHÔNG import SAM → không thiên vị). `eval_handdraw.py` đo lại mọi thứ.

**Định hướng:** Mọi mask cũ đều do SAM hỗ trợ tạo → mọi Dice đo "SAM khớp SAM" = **vòng tự chứng minh**. Phải có ground-truth độc lập để biết số thật.

**PHÁT HIỆN CHẤN ĐỘNG (đây là chỗ sửa lớn nhất của cả dự án):**
- Nhãn SAM cũ vs sự thật chỉ **0.554** (không phải 0.95+). SAM **đếm sót mảnh** ở ca đa-u.
- ⟹ **Mọi Dice 0.958 / 0.965 / 1.000 báo cáo trước đó đều là ẢO** — đo độ khớp với chính nó, không phải khớp sự thật.

**Khắc phục & tại sao:** Từ đây **mọi đánh giá đều dùng 50 ảnh vẽ tay**. Vì chỉ ground-truth độc lập mới phá được vòng tự chứng minh và cho con số tin được.

**Trước/sau:**
| | Dice "báo cáo" trước | Dice thật (vẽ tay) |
|---|---|---|
| Pipeline đầy đủ tự động | ~0.96 (ảo) | **~0.55–0.58** |
Đây là lần "khắc phục" quan trọng nhất: **sửa cách ĐO**, không phải sửa model.

---

## 10. Eval cũ SAI vì "đưa box sẵn" → pipeline end-to-end đúng

**Đã làm:** `e2e_pipeline.py` — mô hình **TỰ** đề xuất box → SAM → so mask tay (không rò rỉ).

**SAI — eval trước đưa sẵn box GT cho SAM:**
- Đưa box đúng thì tất nhiên Dice cao (0.96). Nhưng full-auto thật phải tự tìm box.

**Khắc phục & tại sao:** Đo end-to-end (tự đề xuất box). **Dice tụt còn ~0.55** — lộ ra **localize là nút thắt thật**, bị che bởi cách eval cũ.

**Trước/sau:** box-sẵn 0.96 → tự-đề-xuất 0.55. Chênh 0.41 chính là "thuế localize".

---

## 11. Dense detector (tự chọn nhiều box) — DETR thất bại → FCOS/dense

**Đã làm:** Thiết kế detector tự đề xuất NHIỀU box cho ca đa-u (`detector.py`, `train_detector.py`).

**SAI — DETR-query collapse:**
- Kiểu DETR (query học vị trí) **sụp đổ**: val 0.42 → 0.02 trên 514 ảnh (quá ít mẫu cho query learning).

**Khắc phục & tại sao:** Đổi sang **dense/FCOS** (mỗi ô lưới dự objectness + box). Ổn định, multi-box tự nhiên, tiết kiệm mẫu. → baseline val 0.674.

**Trước/sau:** DETR 0.02 (hỏng) → dense 0.674 (chạy được).

---

## 12. Loạt thử nghiệm cải thiện detector (eval trên 50 vẽ tay)

Mọi cải tiến từ đây đo trên **50 ảnh vẽ tay** (đã có test thật).

| Phiên bản | median | Ghi chú |
|---|---|---|
| dense baseline | ~0.55 | điểm xuất phát |
| + mask-loss (siết box) | 0.581 | tốt trên eval-SAM nhưng **hại** trên vẽ tay (siết box quá → mask thiếu so với vẽ tay rộng) |
| **+ center-sampling** ⭐ | **0.666** | **CHAMPION** — ép nhiều ô gần tâm thành dương → box gọn, bớt thừa |
| grid 128 (lưới mịn) | 0.521 | THUA — lưới mịn sinh nhiều box thừa → SAM nuốt thêm |
| mask-loss + center-sampling | 0.581 | THUA — mask-loss siết box hại trên vẽ tay |
| retrain 162 mask thật | 0.599 | THUA — plateau dữ liệu |

**Định hướng & các SAI lặp lại:** mỗi lần một giả thuyết (siết box / lưới mịn / nhãn thật nhiều hơn). **Bài học chung:** cái tốt trên eval-SAM (mask gọn) lại **hại** trên vẽ tay (annotator vẽ phủ rộng) — phải tin test độc lập, không tin val SAM-GT.

**Khắc phục & tại sao:** Chốt **`detector_recall.pt` (grid64, center-sampling, box-only) = champion 0.666.** Các lever train detector khác đều không vượt → **cạn lever train detector.**

---

## 13. Hậu xử lý inference (gate thước/nhãn + fallback recall) — cải thiện nhỏ

**Đã làm:** `gate_by_specimen()` bỏ box có tâm ngoài bệnh phẩm; `propose_boxes()` — nếu ảnh còn ≤1 box thì giải lại ở ngưỡng thấp hơn (0.35).

**Định hướng:** Cứu ca recall-starved (ít box) mà không hại ca đã đủ box.

**Trước/sau:** median 0.581 → **0.585**, mean 0.529 → 0.563; nhóm đa-u thiếu-box (7 ca) median 0.105 → 0.296 (**≈ gấp đôi**). Đúng nút thắt đa-u.

**SAI tiếp — TTA/ensemble thêm box:** thêm box (lật ảnh, gộp nhiều ckpt) **hại đơn điệu** (0.666 → 0.604 → 0.566 → 0.523). ⟹ nút thắt KHÔNG phải thiếu ứng viên box — hạ bar chỉ thêm RÁC. **Cạn lever inference.**

---

## 14. Phân rã lỗi: localize hay segment? → rồi: box hay "đỏ" (SAM)?

**Đã làm:** Hai lần phân rã lỗi (`scratch_handdraw_diag.py`, `scratch_red_ceiling.py`).

**Phát hiện:**
- **Trần box→mask (box đúng → SAM) = 0.857** (`scratch_deliverable_handdraw.py`, n=50). ⟹ SAM segment TỐT khi box đúng.
- **box-recall (box phủ % u) = 0.88** → detector TÌM TRÚNG u.
- **spill (mask SAM tràn ngoài u) = 0.275** → **SAM tô lố ra mô lành** là điểm yếu, KHÔNG phải box.
- Nếu segment hoàn hảo trong box hiện có → "red-ceiling" = 0.938.

**Kết luận:** khoảng cách 0.666 → 0.857 **không do box** mà do SAM khoanh lố ở box hơi lỏng.

---

## 15. Sửa "đỏ" (SAM): fine-tune decoder & chọn-mask → đều thua

**Đã làm:**
- `finetune_red_truth.py` — FT decoder trên mask thật, box jitter rộng để dạy SAM bám u. → full-auto 0.666 → **0.639** (tệ hơn, overfit tức thì, 166 mask quá ít).
- `scratch_multimask_oracle.py` — SAM trả 3 mask, thử chọn tốt nhất. Oracle = 0.660 ≈ hiện tại 0.666. ⟹ **trong 3 mask SAM không có cái nào bó sát u** — cả 3 đều spill.

**Tại sao thua:** Gốc rễ là **ảnh đại thể: u không tách bạch thị giác với mô lành.** Cho SAM box hơi lỏng → SAM khoanh vùng nổi bật (cả lát), bất kể train hay chọn mask. ⟹ **0.666 là TRẦN THẬT của full-auto.**

---

## 16. Hướng A: segmenter chuyên dụng (bỏ box→SAM) + thử encoder lớn nhất

**Đã làm:** `seg_sam.py` — head U-Net 0.63M trên **đặc trưng SAM encoder đông cứng**, dự đoán DENSE mask u trực tiếp (học u theo kết cấu/màu, bỏ box→SAM).

**Định hướng:** Nếu nút thắt là SAM-spill khi có box, thử cách KHÁC: học u trực tiếp từ feature, không qua box.

**Kết quả + thử "thay model nhỏ → lớn nhất" (yêu cầu mới nhất, đủ GPU):**

| Encoder | median | mean | 1u | >1u |
|---|---|---|---|---|
| SAM tiny | 0.624 | 0.604 | 0.838 | 0.591 |
| **SAM large** | **0.637** | **0.624** | 0.842 | 0.599 |

**SAI — giả định "model lớn cứu được":**
- Large chỉ +0.013 median / +0.020 mean so tiny. **Vẫn thua champion 0.666**, xa trần 0.857; phần đa-u gần như đứng yên.

**Tại sao & khắc phục:** Khớp finding cũ (Bước 7): nút thắt **không phải dung lượng encoder** mà là u không tách thị giác ở ảnh đại thể. DINOv3 7B (26GB) bất khả thi (GPU chỉ ~17GB trống). ⟹ **Giữ nguyên: full-auto trần 0.666; deliverable semi-auto 0.857.**

**Trước/sau:** Trước = kỳ vọng encoder lớn vượt 0.666. Sau = 0.637 < 0.666, xác nhận **lever model đã cạn ở cả hai paradigm** (box→SAM và dense segmenter).

---

## 17. Tổng kết các lần "sai → sửa" và bài học

| # | Sai / giả định | Phát hiện sai nhờ | Khắc phục | Trước → Sau |
|---|---|---|---|---|
| 1 | cm → sinh bbox | phản biện logic | reframe sang box→mask | bế tắc → khả thi |
| 2 | px/cm từ dò thước | Spearman 0.088 | bỏ px/cm, QC mắt | loạn → ổn định |
| 3 | SAM khoanh cả lát = đúng | mask 8cm vs u 3cm | chốt tumor-only | nhãn sai → nhãn đúng |
| 4 | fine-tune > zero-shot | overfit, N=200 thua 2:1 | chốt zero-shot | 0.950 → 0.958 |
| 5 | model lớn > nhỏ (box) | viền trùng khít | giữ tiny | bằng nhau, tiết kiệm 6× GPU |
| 6 | auto-box one-click | Dice 0.33, nổ | prefill-only | 0.33 → người-duyệt |
| 7 | ⭐ Dice 0.95 là thật | **test vẽ tay** | đo lại bằng GT độc lập | 0.96 ảo → 0.55 thật |
| 8 | eval đưa box sẵn | e2e tự đề xuất | eval end-to-end | 0.96 → 0.55 (lộ localize) |
| 9 | DETR-query | val sụp 0.02 | đổi dense/FCOS | 0.02 → 0.674 |
| 10 | mask-loss/grid128/nhãn-nhiều cứu được | đều thua trên vẽ tay | chốt center-sampling | champion 0.666 |
| 11 | thêm box (TTA/ensemble) | hại đơn điệu | giữ box tự-tin | 0.666 cao nhất |
| 12 | FT-đỏ / chọn-mask cứu spill | oracle ≈ hiện tại | chốt 0.666 là trần | xác nhận trần |
| 13 | encoder lớn (seg_sam) | +0.013 không đáng | giữ kết luận | 0.637 < 0.666 |

**Bài học lớn nhất:** **Đầu tư vào CÁCH ĐO (ground-truth độc lập) quan trọng hơn đầu tư vào model.** Suốt nhiều tuần các con số 0.95+ là ảo; chỉ khi có 50 mask vẽ tay mới thấy số thật ~0.55–0.66 và xác định đúng nút thắt (localize + SAM-spill, KHÔNG phải kích thước model).

---

## 18. Trạng thái chốt & deliverable bàn giao

- **Full-auto:** detector_recall (grid64, center-sampling) → box → SAM2.1 zero-shot @1024. Dice **~0.666** trên test vẽ tay (1u ~0.79, đa-u ~0.58). Mọi lever (model lớn, fine-tune, mask-loss, grid mịn, TTA, ensemble, chọn-mask, segmenter chuyên dụng) đã thử & không vượt.
- **Semi-auto (DELIVERABLE):** người duyệt/sửa box prefill (vài giây/ảnh) → SAM → mask **0.857**. Script: `predict_box2mask.py` (+ `README_deliverable.md`).
- **Dataset:** `deliverable_dataset/` = 1020 mask curated + manifest; 50 mask vẽ tay độc lập làm test chuẩn.

**Đòn bẩy duy nhất còn lại để đẩy full-auto:** dữ liệu/nhãn THẬT quy mô lớn hơn cho khâu localize (không phải model to hơn) — nhưng kỳ vọng cận biên thấp vì gốc rễ là u không tách thị giác ở ảnh đại thể.

---

## 19. BEST CURRENT RUN — full-auto specimen-strict trên toàn bộ data

**Ngày chốt:** 2026-06-27.

**Bản chạy tốt nhất cho đến hiện tại để dùng thực tế trên toàn bộ ca:** `specimen_strict` full-auto.

**Lý do chốt:** baseline cũ `detector_recall + SAM2.1 tiny` có Dice tốt hơn nhẹ trên 50 handdraw (`median ~0.6659`) nhưng có lỗi nghiêm trọng: mask có thể tô đè ra background xanh/tím khi box lỏng. Bản `specimen_strict` khóa detector và mask theo vùng specimen, nên phù hợp hơn để chạy toàn bộ data production.

**Lệnh chạy:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/hvusynh2/conda_envs/medsam2_anno/bin/python -u predict_full_auto_all.py --out results/full_auto_specimen_strict_20260626 --overwrite --specimen_strict --min_box_spec_frac 0.50
```

**Output chính:**
- `results/full_auto_specimen_strict_20260626/summary.csv`
- `results/full_auto_specimen_strict_20260626/boxes.csv`
- `results/full_auto_specimen_strict_20260626/masks/`
- `results/full_auto_specimen_strict_20260626/overlays_thumb/`

**Kết quả full-data:**
- Chạy đủ `1393/1393` ca.
- Sinh đủ `1393` mask và `1393` overlay.
- `mask_outside_specimen = 0.0` trên toàn bộ ca → chặn được lỗi tô background.
- `mask/specimen` median `0.3373`, mean `0.4677`.
- `n_box` median `4`, mean `4.43`.
- `7` ca không detect được box.
- `399` ca `mask/specimen < 0.10`; đây là nhóm cần rà tiếp vì khả năng detector/segmentation bỏ sót, không phải lỗi tràn nền.

**Trạng thái chốt:** `results/full_auto_specimen_strict_20260626` là bản full-auto tốt nhất hiện tại theo tiêu chí chạy toàn bộ data không tô nền.

---

## 20. Cập nhật 2026-06-27 — hướng đúng nhất hiện tại: component/slice-aware strict inference

### 20.1. Các thử nghiệm sau baseline 0.666

**Đã thử theo yêu cầu:**
- Focal Loss + Dice Loss trong `seg_crop.py` (`--loss focaldice`).
- Color thresholding / HSV background mask.
- Connected Component Analysis.
- Tăng confidence threshold 0.6/0.7.
- Điều chỉnh NMS 0.3/0.7.
- Morphological erosion/opening.
- Active contour/snaking.
- Post-process clip vào specimen.

**Kết quả chính:**
- `results/fullauto_refinement_eval.csv`
- `results/fullauto_refinement_summary.json`
- Không postprocess nào vượt baseline full-auto cũ trên 50 handdraw.
- Focal+Dice crop segmenter có ceiling tốt khi box đúng (`CEILING median ~0.8861`) nhưng full-auto kém hơn vì nút thắt vẫn là detector/box (`FULL-AUTO median ~0.6036`).

**Bài học:** loss/segmenter giúp nếu box đúng, nhưng không giải quyết lỗi full-auto khi ảnh nhiều lát, box dồn sai lát, hoặc SAM lem trong specimen.

### 20.2. Chẩn đoán lỗi tô background và fix specimen-strict

**Hiện tượng:** có ca mask tô đỏ tràn ra background xanh/tím, ví dụ `SS21-34460`.

**Chẩn đoán:** `clean_specimen()` thực ra tách specimen tốt. Lỗi nằm ở pipeline inference cũ:
- chỉ gate box bằng tâm box nằm trong specimen;
- không mask detector grid theo specimen;
- không reject box có phần lớn diện tích là background;
- không clip mask SAM về specimen.

**Fix đã thêm:**
- `specimen_strict.py`
- `predict_full_auto_all.py --specimen_strict`
- `eval_specimen_strict.py`

**Kết quả full-data đã chạy đủ:** `results/full_auto_specimen_strict_20260626`
- `1393/1393` ca.
- `1393` mask, `1393` overlay.
- `mask_outside_specimen = 0.0` cho toàn bộ ca.
- `mask/specimen` median `0.3373`, mean `0.4677`.
- `n_box` median `4`, mean `4.43`.
- `7` ca không detect box.
- `399` ca `mask/specimen < 0.10`.

**Chốt:** đây là **bản full-data ổn định tốt nhất đã chạy đủ toàn bộ data** vì không còn lỗi tô background.

### 20.3. Kiểm tra max object không bị set nhầm bằng 1

**Kết luận:** không có cấu hình `max_instances`, `max_detections_per_image`, `max_det` nào đang bị set bằng `1`.

**Bằng chứng code:**
- `detector.py`: `decode_detections(..., max_box=20)`.
- `specimen_strict.py`: `decode_detections_specimen(..., max_box=20)`.
- `predict_full_auto_all.py`: không truyền giới hạn về `1`.
- `fallback_if_le=1` trong `propose_boxes()` **không phải giới hạn object**, mà là trigger fallback threshold nếu sau gate còn `<=1` box.

**Bằng chứng output full-data:**
- `results/full_auto_specimen_strict_20260626/summary.csv` có `n_box` từ `0` đến `15`.
- `boxes.csv` có tổng `6177` box.
- Có `295` ảnh chỉ có 1 box, nhưng đó là do detector/threshold/specimen filter, không phải do hard cap.

### 20.4. Vấn đề còn lại: ảnh nhiều lát cắt

**Triệu chứng mới:** bản hiện tại chạy tốt, nhưng với ảnh có nhiều lát cắt:
- Có lát bị tô lem vào mô không tổn thương.
- Có lát có tổn thương nhưng không được detect.
- Ví dụ rõ: nhóm `SS21-38576`, nhiều lát nằm trên cùng ảnh.

**Phân tích:** đây không còn là lỗi background. Đây là lỗi **thiếu slice-awareness**:
- `specimen_strict` xem toàn bộ specimen union là một vùng lớn.
- Detector có thể dồn nhiều box vào một lát có tín hiệu mạnh.
- Các lát khác không được quota/attention riêng.
- SAM mask chỉ clip vào toàn specimen, chưa clip theo từng lát/component.

### 20.5. Hướng đúng nhất hiện tại: component/slice-aware strict

**Ý tưởng:** tách specimen thành các connected components lớn, coi mỗi component là một lát cắt riêng:
1. Tách background bằng `clean_specimen()`.
2. Tách từng lát bằng connected components.
3. Decode box riêng trong từng component.
4. Không ép mỗi component phải có box nếu score không vượt threshold.
5. SAM predict từng box.
6. Clip mask của box về đúng component/lát chứa box.
7. Gộp mask các lát.

**Code đã thêm:**
- `component_strict.py`
- `eval_component_strict.py`
- `predict_full_auto_all.py --component_strict`

**Output thử nghiệm:**
- `results/component_strict_eval/component_strict_eval.csv`
- `results/component_strict_eval/component_strict_summary.json`
- `results/component_strict_eval/demo_overlays/`
- `results/component_strict_eval/SS21-38576_strict_vs_component_contact.jpg`
- `results/component_strict_demo_SS21_38576/`
- `results/component_strict_demo_SS21_38576_comp2/`

**Kết quả trên 50 handdraw:**
```text
strict050              median=0.6434 mean=0.5930 1u=0.7630 >1u=0.5499 box_med=4 mask/spec=0.310
comp4_frac045_fb025    median=0.6408 mean=0.6032 1u=0.7629 >1u=0.5871 box_med=4 mask/spec=0.273
comp3_frac050_fb025    median=0.6390 mean=0.5839 1u=0.7325 >1u=0.5508 box_med=3 mask/spec=0.235
comp2_frac050_fb035    median=0.6203 mean=0.5736 1u=0.7370 >1u=0.5502 box_med=2 mask/spec=0.195
comp1_frac050_fb035    median=0.5030 mean=0.4783 1u=0.5953 >1u=0.4914 box_med=1 mask/spec=0.123
```

**Diễn giải:**
- `comp4_frac045_fb025` là **hướng thử nghiệm tốt nhất hiện tại cho ảnh nhiều lát/nhiều u**: mean tăng `0.5930 -> 0.6032`, multi-u tăng `0.5499 -> 0.5871`, mask/spec giảm `0.310 -> 0.273`.
- Median hơi thấp hơn strict (`0.6408` vs `0.6434`), nên chưa thay thế bản full-data đã chốt nếu tiêu chí là median 50 handdraw.
- `comp2_frac050_fb035` ít lem hơn (`mask/spec=0.195`) nhưng giảm recall/median; phù hợp nếu ưu tiên ít tô nhầm hơn bắt đủ.

**Demo `SS21-38576`:**
- `specimen_strict` thường dồn box vào một component/lát.
- `component_strict` phân phối box sang nhiều lát khác nhau.
- Contact sheet xác nhận hướng slice-aware đúng với lỗi user đang thấy.

### 20.6. Lệnh chạy tiếp

**Bản full-data ổn định tốt nhất hiện tại đã chạy xong:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/hvusynh2/conda_envs/medsam2_anno/bin/python -u predict_full_auto_all.py --out results/full_auto_specimen_strict_20260626 --overwrite --specimen_strict --min_box_spec_frac 0.50
```

**Bản nên chạy tiếp để kiểm chứng hướng đúng nhất hiện tại trên toàn bộ data (`component_strict comp4`):**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/hvusynh2/conda_envs/medsam2_anno/bin/python -u predict_full_auto_all.py --out results/full_auto_component_strict_comp4_20260627 --overwrite --component_strict --component_max_box 4 --component_min_box_spec_frac 0.45 --component_fallback_thr 0.25
```

**Đã chạy xong full-data cho `component_strict comp4`:**
- Output: `results/full_auto_component_strict_comp4_20260627/`
- Summary: `results/full_auto_component_strict_comp4_20260627/summary.csv`
- Boxes: `results/full_auto_component_strict_comp4_20260627/boxes.csv`
- Masks: `results/full_auto_component_strict_comp4_20260627/masks/`
- Overlays: `results/full_auto_component_strict_comp4_20260627/overlays_thumb/`
- Chạy đủ `1393/1393` ca.
- Có `1393` mask và `1393` overlay.
- `boxes.csv` có `6408` box data rows.
- `n_box` median `4`, mean `4.60`, max `23`, zero-box `12`.
- `mask/specimen` median `0.3503`, mean `0.4677`.
- `mask_outside_specimen = 0.0` toàn bộ ca.
- `mask/specimen <0.05`: `116` ca.
- `mask/specimen <0.10`: `261` ca.

**So với `specimen_strict` full-data:**
- `mask/specimen` median tăng `0.3373 -> 0.3503`.
- `mask/specimen <0.10` giảm `399 -> 261` ca, tức giảm nhóm mask quá nhỏ/bỏ sót.
- `n_box` mean tăng `4.43 -> 4.60`, đúng kỳ vọng vì chia theo lát.
- zero-box tăng `7 -> 12`, vì component decoder không ép top-1 nếu score không vượt ngưỡng.
- `mask_outside_specimen` vẫn bằng `0.0`.

**Trạng thái sau run:** `component_strict comp4` là bản full-data tốt nhất hiện tại cho hướng slice-aware/nhiều lát. Cần QC overlay ở các ca nhiều lát để xác nhận tăng recall không đổi lấy quá nhiều tô lem nội-specimen.

**Bản ít lem hơn nếu cần conservative (`component_strict comp2`):**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/hvusynh2/conda_envs/medsam2_anno/bin/python -u predict_full_auto_all.py --out results/full_auto_component_strict_comp2_20260627 --overwrite --component_strict --component_max_box 2 --component_min_box_spec_frac 0.50 --component_fallback_thr 0.35
```

**Đã chạy xong full-data cho `component_strict comp2` ngày 2026-06-28:**
- Output gốc: `results/full_auto_component_strict_comp2_20260628/`
- Đã copy sang submission: `/home/hvusynh2/nguyenduong/medsam2_rcc_submission_20260627/results/original/full_auto_component_strict_comp2_20260628/`
- Summary: `results/full_auto_component_strict_comp2_20260628/summary.csv`
- Boxes: `results/full_auto_component_strict_comp2_20260628/boxes.csv`
- Masks: `results/full_auto_component_strict_comp2_20260628/masks/`
- Overlays: `results/full_auto_component_strict_comp2_20260628/overlays_thumb/`
- Chạy đủ `1393/1393` ca.
- Có `1393` mask và `1393` overlay.
- `boxes.csv` có `3242` box data rows.
- `n_box` median `2`, mean `2.33`, max `10`, zero-box `64`.
- `mask/specimen` median `0.2247`, mean `0.3567`.
- `mask_outside_specimen = 0.0` toàn bộ ca.
- `mask/specimen <0.05`: `252` ca.
- `mask/specimen <0.10`: `425` ca.

**Diễn giải `comp2`:** đây là bản conservative/ít lem hơn nhưng bỏ sót nhiều hơn `comp4`. Dùng để so QC nếu `comp4` tô nội-specimen quá rộng; không nên coi là bản recall tốt nhất.

### 20.7. Trạng thái chốt hiện tại

- **Best stable conservative full-data run:** `results/full_auto_specimen_strict_20260626`.
- **Best current slice-aware full-data run:** `results/full_auto_component_strict_comp4_20260627`.
- **Best current technical direction:** `component_strict` / slice-aware strict inference.
- **Best balanced config:** `comp4_frac045_fb025`, đã chạy full-data.
- **Best conservative config:** `comp2_frac050_fb035`, đã chạy full-data và đã copy sang submission.
- **Không nên quay lại:** chỉ tăng model, chỉ thêm postprocess global, hoặc chỉ tăng/giảm confidence toàn ảnh. Những hướng này không xử lý bản chất ảnh nhiều lát.
- **Việc nên làm kế tiếp:** QC overlay của `component_strict comp4` và `component_strict comp2` ở nhóm nhiều lát; chọn `comp4` nếu cần recall, chọn `comp2` nếu ưu tiên giảm tô lem.
