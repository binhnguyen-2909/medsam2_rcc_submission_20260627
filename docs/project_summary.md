# Project Summary

Task: tumor segmentation on kidney RCC gross-pathology photos.

Data facts:

- Raw images are camera photos of resected kidney cut surfaces.
- The spreadsheet tumor size describes lesion size, not specimen size and not
  tumor location.
- Ruler-derived px/cm was unreliable and is not used as the primary quality
  signal.

Current conclusion:

- Full-auto localization is capped around median Dice 0.64 on the hand-drawn
  test set.
- Semi-auto tight box-to-mask is the reliable deliverable.
- The current best semi-auto segmenter is `SegResNet+LAB` on cropped boxes,
  median Dice 0.883 on 50 independent hand-drawn masks.

Primary deliverable:

```bash
python src/inference/predict_seg_crop.py --config configs/model/seg_crop_lab.yaml ...
```
