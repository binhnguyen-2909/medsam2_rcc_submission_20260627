# Rejected Experiment Directions

Mục tiêu của file này là trả lời câu hỏi: mọi hướng đã từng chạy rồi bị bác bỏ có được ghi lại đủ để người khác hiểu và không lặp lại sai lầm không?

Trạng thái sau audit: **đã bổ sung README tổng hợp này**. Trước đó, `docs/original/TONG_KET_HANH_TRINH.md` có giải thích nhiều hướng, nhưng thiếu các ablation cuối và thiếu format thống nhất cho từng hướng bị loại.

Format mỗi mục:

- Lệnh/script: script hoặc runner đã chạy.
- Thông số: cấu hình chính.
- Cơ sở lý thuyết: tại sao thử hướng này.
- Kết quả: số chính hoặc file kết quả.
- Phân tích: tại sao bị loại và chuyển sang hướng nào.

Các log/bảng gốc nằm trong `results/original/`, script gốc trong `src/legacy/`, runner gốc trong `scripts/legacy/`, checkpoint trong `checkpoints/`.

## Baseline Được Giữ Lại

Các hướng dưới đây là mốc so sánh, không phải hướng bị loại:

- Semi-auto deliverable hiện tại: `src/inference/predict_seg_crop.py`, config `configs/model/seg_crop_lab.yaml`, checkpoint `checkpoints/seg_crop_segR_lab.pt`. Khi box đúng, median Dice handdraw50 `0.8834`.
- Full-auto stable full-data: `predict_full_auto_all.py --specimen_strict`, output `results/original/full_auto_specimen_strict_20260626/`. Chọn để chạy toàn bộ data vì không tô ra background.
- Hướng kỹ thuật full-auto tốt nhất để thử tiếp: `component_strict` slice-aware. Chưa thay hẳn stable full-data vì median handdraw chưa vượt, nhưng cải thiện mean/multi-u.

## 1. Excel cm -> Bbox

- Lệnh/script: không có script hoàn chỉnh, bị bác ở mức thiết kế.
- Thông số: dùng kích thước u trong `RCC 20241212.xlsx` để suy ra bbox.
- Cơ sở lý thuyết: nếu Excel có kích thước u, có thể dùng nó làm ràng buộc hình học.
- Kết quả: không thể triển khai đúng vì kích thước không chứa vị trí.
- Phân tích: số cm trong Excel là kích thước khối u, không cho biết u nằm ở đâu trong ảnh. Hướng này bị loại. Chuyển sang bài toán đúng hơn: người hoặc model đưa box, sau đó box-to-mask.

## 2. Px/cm Từ Thước Làm QC Chính

- Lệnh/script: `src/legacy/preprocess.py`, `src/legacy/area_sanity.py`, runner `scripts/legacy/run_all.sh`.
- Thông số: ruler autocorrelation, so diện tích mask quy đổi cm2 với Excel tumor cm2.
- Cơ sở lý thuyết: nếu px/cm đúng, diện tích mask u phải tương quan với kích thước u trong Excel.
- Kết quả: Spearman chỉ `0.088`, nhiều ca lệch lớn do bắt nhầm vạch mm/cm.
- Phân tích: px/cm không đủ tin cậy để làm QC chính. Hướng bị loại cho quyết định model, chỉ giữ như metadata phụ. Chuyển sang QC mắt thường và handdraw test.

## 3. MedSAM2 Là Backbone Chính

- Lệnh/script: `src/legacy/benchmark_backbones.py`, runner `scripts/legacy/compare_backbones.sh`, log `results/original/backbone_benchmark_1024.csv`, `results/original/medsam_cmp.log`.
- Thông số: MedSAM2 latest vs SAM2.1 vanilla tiny/large, box prompt.
- Cơ sở lý thuyết: model y khoa chuyên biệt có thể tốt hơn SAM vanilla.
- Kết quả: MedSAM2 kém hơn trên ảnh đại thể màu. Memory ghi MedSAM2 ceiling khoảng `0.841` và full-auto `0.615`, thấp hơn SAM2.1 và SegResNet+LAB.
- Phân tích: MedSAM2 lệch domain vì ưu tiên ảnh y khoa/xám, không hợp gross pathology RGB. Bị loại. Chuyển sang SAM2.1 vanilla và sau đó SegResNet crop.

## 4. Fine-tune SAM2 Prompt/Mask Decoder

- Lệnh/script: `src/legacy/finetune_sam2.py`, `scripts/legacy/loop_round.sh`, log `results/original/finetune_log.csv`, `results/original/confirm200_ft_vs_zs.csv`.
- Thông số: đóng băng image encoder, train prompt_encoder + mask_decoder, loss Dice+BCE, 47/170/1008 train, test frozen.
- Cơ sở lý thuyết: fine-tune trên domain RCC sẽ vượt zero-shot.
- Kết quả: trên test nhỏ có nhiễu, nhưng xác nhận N=200 cho thấy zero-shot median `0.9645` > FT `0.9502`, FT thua 132/200.
- Phân tích: fine-tune học lại nhãn SAM-assisted và overfit. Bị loại cho deliverable box-to-mask. Chuyển sang zero-shot SAM baseline và sau đó test handdraw độc lập.

## 5. SAM2.1 Large Thay Tiny

- Lệnh/script: `src/legacy/eval_zeroshot.py`, `scripts/legacy/compare_backbones.sh`, kết quả `results/original/zeroshot_large.json`, `results/original/zeroshot_tiny.json`, `results/original/tiny_vs_large.jpg`.
- Thông số: SAM2.1 tiny vs large, image size 1024.
- Cơ sở lý thuyết: backbone lớn hơn có thể cho biên mask tốt hơn.
- Kết quả: tiny không thua large trong box-to-mask; large tốn GPU hơn nhiều. Với `seg_sam`, large chỉ tăng từ median `0.624` lên `0.637`, vẫn dưới full-auto champion.
- Phân tích: nút thắt không phải dung lượng encoder. Bị loại. Chuyển sang cải thiện nhãn/eval/localization.

## 6. AMG / Automatic Mask Proposal Một Cú Bấm

- Lệnh/script: `src/legacy/propose_amg.py`, `src/legacy/amg_classify.py`, log `results/original/amg_demo.jpg`, `results/original/loc_amg.json`.
- Thông số: grid-point SAM proposal, NMS, sau đó classifier proposal.
- Cơ sở lý thuyết: sinh nhiều mask tự động rồi chọn mask giống u.
- Kết quả: proposal chủ yếu là mảnh mô rời rạc, không biết u ở đâu. Localizer AMG median `0.356`.
- Phân tích: máy không có tín hiệu thị giác đủ rõ để phân biệt u/lành. Bị loại làm full-auto. Chuyển sang prefill-only và detector học có giám sát.

## 7. Auto-box -> SAM One-click

- Lệnh/script: `src/legacy/train_cellbox.py`, `src/legacy/propose_box.py`, `src/legacy/scratch_eval_autobox.py`, `src/legacy/scratch_eval_autobox2.py`.
- Thông số: cell classifier trên lưới 30 ô, vote threshold khoảng `0.6-0.7`, SAM từ auto box.
- Cơ sở lý thuyết: box gợi ý tự động có thể thay người vẽ box.
- Kết quả: auto-box -> SAM median Dice khoảng `0.22-0.33`, nhiều ca mask nổ hoặc Dice `0`.
- Phân tích: box hơi lỏng làm SAM nuốt mô lành/cả lát. Bị loại làm one-click. Giữ lại chỉ như prefill cho người duyệt.

## 8. Đánh Giá Bằng SAM-made GT

- Lệnh/script: các eval cũ như `src/legacy/eval_zeroshot.py`, `src/legacy/scratch_eval_random.py`, `src/legacy/scratch_compare550.py`.
- Thông số: so prediction với masks do SAM hỗ trợ tạo.
- Cơ sở lý thuyết: dùng nhãn đang có để đo tự động.
- Kết quả: Dice 0.95-1.00 nhưng bị handdraw test bác bỏ. Nhãn SAM cũ vs handdraw chỉ khoảng `0.554`.
- Phân tích: đây là vòng tự chứng minh. Hướng đánh giá này bị loại cho kết luận chất lượng. Chuyển sang `labels_handdraw/masks` làm test chuẩn.

## 9. Eval Đưa Sẵn Box GT Cho Full-auto

- Lệnh/script: `src/legacy/predict_box2mask.py`, `src/legacy/e2e_pipeline.py`, `results/original/e2e_ft_vs_zs.csv`.
- Thông số: box GT hoặc prompt đã lưu vs pipeline tự đề xuất box.
- Cơ sở lý thuyết: đo riêng chất lượng segmenter trong box.
- Kết quả: box-sẵn cho Dice cao nhưng không phản ánh full-auto. Khi tự đề xuất box, Dice tụt mạnh còn khoảng `0.55`.
- Phân tích: cách eval box-sẵn bị loại cho full-auto. Giữ lại chỉ để đo ceiling của box-to-mask.

## 10. DETR-query Detector

- Lệnh/script: bản đầu của `src/legacy/train_detector.py`, sau đó thay bằng `src/model/detector.py` dense FCOS.
- Thông số: set-prediction query matching.
- Cơ sở lý thuyết: DETR-style tự dự đoán nhiều object.
- Kết quả: query collapse, val giảm mạnh, memory ghi khoảng `0.42 -> 0.02`.
- Phân tích: dữ liệu ít, query learning không ổn định. Bị loại. Chuyển sang dense/FCOS grid detector.

## 11. Detector Mask-loss, Grid128, Retrain Truth

- Lệnh/script: `src/legacy/train_detector.py`, logs `results/original/detector_maskloss_train.log`, `detector_recall128_train.log`, `detector_truth_train.log`, eval `results/original/handdraw_eval_*.csv`.
- Thông số: mask loss end-to-end, grid 128, train thêm nhãn thật.
- Cơ sở lý thuyết: siết box, lưới mịn hơn, hoặc nhãn thật hơn sẽ cải thiện localize.
- Kết quả: không vượt `detector_recall.pt` center-sampling. Grid128 và mask-loss làm nhiều ca kém hơn.
- Phân tích: các biến thể này overfit hoặc sinh box thừa/thiếu. Bị loại. Chốt detector_recall grid64 center-sampling.

## 12. TTA / Ensemble Thêm Box

- Lệnh/script: `src/legacy/scratch_tta_ensemble.py`, kết quả `results/original/tta_ensemble.csv`.
- Thông số: flip/test-time augmentation, gộp thêm box từ nhiều biến thể.
- Cơ sở lý thuyết: thêm candidate box sẽ tăng recall, nhất là đa-u.
- Kết quả: thêm box hại đơn điệu theo memory, champion `0.666` giảm còn khoảng `0.604`, `0.566`, `0.523`.
- Phân tích: vấn đề không phải thiếu box mà là box rác làm SAM/segmenter tô thêm mô lành. Bị loại.

## 13. Segmenter Trực Tiếp Trên SAM Feature: seg_sam

- Lệnh/script: `src/legacy/seg_sam.py`, `scripts/legacy/run_seg_sam.sh`, `run_seg_sam_large.sh`, kết quả `results/original/seg_sam_handdraw.json`, `seg_sam_handdraw_large.json`.
- Thông số: U-Net head 0.63M trên SAM encoder features, tiny/large.
- Cơ sở lý thuyết: bỏ box-prompt và học mask u trực tiếp từ feature.
- Kết quả: tiny median `0.624`, large `0.637`, đều thấp hơn full-auto detector+SAM khoảng `0.666`.
- Phân tích: feature SAM không đủ tách u/lành trong ảnh đại thể. Bị loại. Chuyển sang crop segmenter chuyên dụng.

## 14. Crop Segmenter Ablation: RGB, LAB texture, Boundary, Swin

- Lệnh/script: `src/legacy/seg_crop.py`, `scripts/legacy/run_seg_crop.sh`, kết quả `results/original/seg_crop_*.json`.
- Thông số: SegResNet/SwinUNETR, RGB/LAB/LAB+Gabor, DiceBCE/Boundary, crop pad 0.15, size 512.
- Cơ sở lý thuyết: thay SAM trong box bằng segmenter chuyên dụng để giảm spill.
- Kết quả:
  - `segR_lab` được giữ: ceiling median `0.8834`.
  - `segR_rgb`: full-auto `0.616`, ceiling `0.879`.
  - `segR_labtex`: full-auto `0.650`, ceiling `0.874`.
  - `segR_labtex_bd`: full-auto `0.643`, ceiling `0.878`.
  - `swin_labtex_bd`: full-auto `0.627`, ceiling `0.867`.
- Phân tích: LAB SegResNet là lựa chọn cân bằng nhất. RGB, LAB texture, boundary loss và Swin bị loại vì không vượt champion deliverable hoặc làm full-auto/ceiling kém hơn.

## 15. Focal Dice Cho Crop Segmenter

- Lệnh/script: `src/legacy/seg_crop.py --loss focaldice`, kết quả `results/original/seg_crop_segR_labtex_focaldice.json`.
- Thông số: LAB+texture, focal+dice.
- Cơ sở lý thuyết: xử lý mất cân bằng foreground/background tốt hơn.
- Kết quả: ceiling median `0.8861` nhưng full-auto median chỉ `0.6036`, thấp hơn `segR_lab` full-auto và không đủ cải thiện deliverable một cách rõ ràng.
- Phân tích: loss có thể giúp khi box đúng nhưng không giải quyết localize. Bị loại cho pipeline chính.

## 16. Adversarial PatchGAN, clDice, Adv+clDice

- Lệnh/script: `src/legacy/seg_crop_adv.py`, `scripts/legacy/run_seg_adv.sh`, kết quả `results/original/seg_crop_segR_lab_adv*.json`.
- Thông số: PatchGAN LSGAN lambda 0.05, clDice lambda 0.5.
- Cơ sở lý thuyết: ép mask có hình thái giống nhãn thật, siết biên/spill.
- Kết quả: clDice có ceiling cao nhất cục bộ khoảng `0.8925` nhưng paired gain rất nhỏ và nhiễu; adversarial giảm hiệu năng.
- Phân tích: 318 patch quá ít, mask thật cũng mang phong cách annotation không ổn định. Không đổi deliverable vì lợi ích không vững. Bị loại.

## 17. MAE Self-supervised Pretrain

- Lệnh/script: `src/legacy/mae_seg.py`, `scripts/legacy/run_mae.sh`, kết quả `results/original/seg_crop_seg_mae.json`, `seg_crop_seg_unet_scratch.json`.
- Thông số: MAE trên 1128 cut-surface, finetune SmallUNet; đối chứng scratch.
- Cơ sở lý thuyết: pretrain tự giám sát trên domain ảnh sẽ học texture u/lành.
- Kết quả: MAE khoảng `0.884`, scratch khoảng `0.885`, ngang nhau.
- Phân tích: pretrain không thêm tín hiệu phân biệt u/lành. Bị loại.

## 18. FFT Two-stream

- Lệnh/script: `src/legacy/seg_crop_fft.py`, `scripts/legacy/run_seg_new.sh`, kết quả `results/original/seg_crop_fft_lab.json`.
- Thông số: RGB/LAB branch + FFT log magnitude branch.
- Cơ sở lý thuyết: u có thể khác mô lành ở miền tần số/texture.
- Kết quả: ceiling khoảng `0.866`, kém `segR_lab`.
- Phân tích: thêm tham số gây overfit, tín hiệu frequency không đủ. Bị loại.

## 19. Body-edge Decoupling

- Lệnh/script: `src/legacy/seg_crop_be.py`, `scripts/legacy/run_seg_new.sh`, kết quả `results/original/seg_crop_be_lab.json`.
- Thông số: head body eroded mask + edge ring.
- Cơ sở lý thuyết: tách thân u và biên giúp rõ boundary.
- Kết quả: ceiling khoảng `0.860`, kém `segR_lab`.
- Phân tích: biên u/lành mờ và chủ quan, head tách biên overfit. Bị loại.

## 20. Postprocess Guided Filter / Snakes / Active Contour

- Lệnh/script: `src/legacy/refine_postproc.py`, `src/legacy/eval_fullauto_refinements.py`, kết quả `results/original/refine_postproc.json`, `fullauto_refinement_summary.json`.
- Thông số: guided filter, GAC, ACWE, morphology, threshold/NMS variants.
- Cơ sở lý thuyết: hậu xử lý có thể co mask về biên thật.
- Kết quả: guided gần trung tính; GAC/ACWE hại ceiling; active contour full-auto không vượt baseline.
- Phân tích: biên u/lành yếu, snake snap vào cạnh specimen hoặc nhiễu. Bị loại.

## 21. Localizer Box Refiner

- Lệnh/script: `src/legacy/box_refiner.py`, `src/legacy/localize_eval.py --method refiner`, kết quả `results/original/loc_refiner.json`.
- Thông số: CNN siết box từ jitter GT.
- Cơ sở lý thuyết: học biến box lỏng thành box khít.
- Kết quả: median `0.630`, thấp hơn detector `0.635`.
- Phân tích: jitter mô phỏng không giống lỗi detector thật. Bị loại.

## 22. YOLOv11 Localizer

- Lệnh/script: `src/legacy/train_yolo.py`, `src/legacy/localize_eval.py --method yolo`, kết quả `results/original/loc_yolo.json`.
- Thông số: YOLOv11s, dataset `data/interim/yolo_ds`, checkpoint `checkpoints/yolo_best.pt`.
- Cơ sở lý thuyết: detector object chuyên dụng có thể localize u tốt hơn dense head.
- Kết quả: mAP50 khoảng `0.594`, nhưng mask pipeline median `0.5606`.
- Phân tích: mAP box không chuyển thành Dice tốt; box vẫn sai/lỏng theo cách làm segmenter hỏng. Bị loại.

## 23. SLIC Superpixel Localizer

- Lệnh/script: `src/legacy/slic_clf.py`, `localize_eval.py --method slic`, kết quả `results/original/loc_slic.json`.
- Thông số: SLIC superpixel + MLP màu/texture/vị trí.
- Cơ sở lý thuyết: superpixel theo biên tự nhiên có thể gom vùng u.
- Kết quả: median khoảng `0.495`.
- Phân tích: u và mô lành cùng màu/texture, đa-u sụp. Bị loại.

## 24. Grid CNN Patch Classifier

- Lệnh/script: `src/legacy/grid_clf.py`, `localize_eval.py --method grid`, kết quả `results/original/loc_grid.json`.
- Thông số: patch/grid heatmap -> box.
- Cơ sở lý thuyết: học heatmap vị trí u trực tiếp.
- Kết quả: median khoảng `0.505`.
- Phân tích: 1-u đôi khi ổn nhưng đa-u kém, tín hiệu u/lành chồng lấn. Bị loại.

## 25. Centerpoint / Point Prompt

- Lệnh/script: `localize_eval.py --method centerpoint`, kết quả `results/original/loc_centerpoint.json`.
- Thông số: detector center -> SAM point-prompt.
- Cơ sở lý thuyết: point prompt ít bị box lỏng làm spill.
- Kết quả: median khoảng `0.599`, thấp hơn detector box route.
- Phân tích: point prompt thiếu ràng buộc kích thước/extent. Bị loại.

## 26. Size Constraint Từ Excel

- Lệnh/script: `localize_eval.py --method size`, kết quả `results/original/loc_size.json`.
- Thông số: lọc box bằng tumor cm2 Excel và px/cm metadata.
- Cơ sở lý thuyết: kích thước u có thể loại box sai.
- Kết quả: median không đổi khoảng `0.635`.
- Phân tích: lỗi chính là vị trí/ranh giới, không phải size; px/cm không tin. Bị loại.

## 27. Anomaly Autoencoder

- Lệnh/script: `src/legacy/anomaly_ae.py`, `localize_eval.py --method anomaly`, kết quả `results/original/loc_anomaly.json`.
- Thông số: ConvAE train patch mô lành, dùng reconstruction error heatmap.
- Cơ sở lý thuyết: u là bất thường so với mô lành.
- Kết quả: median khoảng `0.057`.
- Phân tích: u không phải anomaly thị giác rõ; reconstruction error bắt nhiễu/texture khác. Bị loại.

## 28. Mask2Former-lite

- Lệnh/script: `src/legacy/mask2former_lite.py`, `scripts/legacy/run_m2f.sh`, kết quả `results/original/loc_mask2former.json`.
- Thông số: Mask2Former-lite localizer/segmenter thử nghiệm.
- Cơ sở lý thuyết: architecture segmentation hiện đại có thể học mask tốt hơn head nhẹ.
- Kết quả: median `0.5430`, mean `0.5285`.
- Phân tích: không vượt detector/segR route, dữ liệu ít và biên mờ. Bị loại.

## 29. Ellipse Shape Prior

- Lệnh/script: `src/legacy/blob_redceiling.py`, `src/legacy/ellipse_reg.py`, kết quả `results/original/blob_redceiling.json`, `loc_ellipse_reg.json`.
- Thông số: ellipse from GT, ellipse regression from crop.
- Cơ sở lý thuyết: u gần ellipse, shape prior có thể tăng Dice.
- Kết quả: ellipse_of_GT ceiling khoảng `0.949`, nhưng ellipse_reg full-auto khoảng `0.579`, ceiling `0.850`.
- Phân tích: shape prior đúng nếu biết ellipse, nhưng dự đoán ellipse từ ảnh khó như segment. Bị loại.

## 30. Blob Rebox / Blob Merge

- Lệnh/script: `src/legacy/blob_rebox.py`, `src/legacy/blob_merge.py`, kết quả `results/original/blob_rebox.json`, `blob_merge.json`.
- Thông số: rebox từ mask/blob, gộp box chồng.
- Cơ sở lý thuyết: mask hoặc nhiều box có thể sinh ngược box gọn hơn.
- Kết quả: rebox/merge đều hại, ví dụ blob_merge giảm từ khoảng `0.635` xuống `0.608`.
- Phân tích: nhiều box chật tốt hơn một box gộp lỏng; rebox từ mask sai chỉ khuếch đại lỗi. Bị loại.

## 31. VLM Critic

- Lệnh/script: `src/legacy/vlm_eval.py`, `scripts/legacy/run_vlm.sh`, kết quả `results/original/vlm_critic.json`.
- Thông số: VLM/critic chọn hoặc đánh giá vùng.
- Cơ sở lý thuyết: mô hình thị giác-ngôn ngữ có thể nhận biết u bằng ngữ nghĩa.
- Kết quả: VLM median `0.2986`, detector median trong cùng subset `0.6549`.
- Phân tích: VLM không có chuyên môn/độ phân giải phù hợp cho ranh giới u gross pathology. Bị loại.

## 32. Test-time Entropy Minimization

- Lệnh/script: `src/legacy/tent_eval.py`, kết quả `results/original/tent_eval.json`.
- Thông số: TENT 10 iter/crop trên SegResNet GroupNorm.
- Cơ sở lý thuyết: adaptation lúc test có thể làm prediction sắc hơn.
- Kết quả: ceiling `0.8766`, full-auto `0.6397`, trung tính hoặc thấp hơn champion.
- Phân tích: entropy minimization không thêm thông tin ranh giới; GroupNorm hạn chế adaptation. Bị loại.

## 33. SupCon / Contrastive Texture

- Lệnh/script: `src/legacy/supcon_loc.py`, kết quả `results/original/supcon_auc.json`.
- Thông số: dense contrastive InfoNCE phân biệt patch u/lành.
- Cơ sở lý thuyết: ép embedding tách u/lành để localize tốt hơn.
- Kết quả: AUC chỉ `0.7637`.
- Phân tích: đây là bằng chứng định lượng rằng texture u/lành chồng lấn mạnh. Hướng contrastive không đủ. Bị loại như giải pháp, giữ lại như bằng chứng giới hạn dữ liệu.

## 34. DINOv3 7B / Backbone Rất Lớn

- Lệnh/script: không chạy full do tài nguyên; ghi trong memory và log thảo luận.
- Thông số: DINOv3 7B khoảng 26GB checkpoint, GPU trống không đủ ổn định.
- Cơ sở lý thuyết: foundation encoder lớn hơn có thể tách texture tốt hơn.
- Kết quả: không triển khai full, và các thử nghiệm SAM large/SegSAM large đã cho thấy backbone lớn không phải đòn bẩy.
- Phân tích: loại vì không khả thi tài nguyên và giả thuyết đã bị bác gián tiếp. Không ưu tiên.

## 35. Specimen Strict Variants

- Lệnh/script: `src/legacy/specimen_strict.py`, `src/legacy/eval_specimen_strict.py`, `src/legacy/predict_full_auto_all.py --specimen_strict`, kết quả `results/original/specimen_strict_summary.json`.
- Thông số: clip/gate mask theo specimen, threshold 0.35/0.50/0.65, shrink/no-shrink.
- Cơ sở lý thuyết: chặn lỗi tô background xanh/tím.
- Kết quả: strict variants giảm background; median handdraw thấp hơn baseline một chút, nhưng stable full-data không còn outside-specimen.
- Phân tích: không bị loại toàn bộ. Các biến thể threshold cao/thấp, shrink/no-shrink không thắng balanced config nên bị loại. `specimen_strict` được giữ cho full-data stable.

## 36. Component Strict Conservative Variants

- Lệnh/script: `src/legacy/component_strict.py`, `src/legacy/eval_component_strict.py`, `predict_full_auto_all.py --component_strict`, kết quả `results/original/component_strict_eval/component_strict_summary.json`.
- Thông số: `comp1`, `comp2`, `comp3`, `comp4`, min box spec frac, fallback threshold.
- Cơ sở lý thuyết: xử lý ảnh nhiều lát bằng quota/component.
- Kết quả: `comp4_frac045_fb025` tăng mean/multi-u nhưng median chưa vượt strict. `comp1/comp2/comp3` giảm recall hoặc median.
- Phân tích: component-strict không bị loại như hướng, nhưng các config conservative bị loại làm default. Giữ `comp4` là hướng thử tiếp, giữ `specimen_strict` là stable full-data.

## Quy Tắc Không Lặp Lại

1. Không dùng Dice đo trên SAM-made mask để kết luận chất lượng thật.
2. Không dùng Excel cm để suy ra vị trí u.
3. Không tăng model/backbone trước khi chứng minh lỗi do capacity.
4. Không thêm box hàng loạt nếu không có cơ chế reject chắc, vì box rác làm spill.
5. Không tin mAP box nếu metric cuối là mask Dice trên handdraw.
6. Mọi hướng mới phải báo riêng:
   - box đúng -> mask ceiling,
   - full-auto end-to-end,
   - 1-u vs multi-u,
   - lỗi outside specimen,
   - và paired delta với `segR_lab` hoặc full-auto stable baseline.

