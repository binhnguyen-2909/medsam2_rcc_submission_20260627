# Checkpoints

This folder contains model weights required to reproduce the submitted
experiments. Binary weights are tracked with Git LFS.

## Current Deliverable

Use this checkpoint for the champion semi-auto box-to-mask workflow:

```text
checkpoints/seg_crop_segR_lab.pt
```

with:

```text
configs/model/seg_crop_lab.yaml
src/inference/predict_seg_crop.py
```

Independent hand-drawn evaluation for this deliverable is summarized in
`results/tables/seg_crop_segR_lab.json`.

## Important Baseline Weights

- `detector_recall.pt`: best retained dense detector/localizer baseline.
- `sam2.1_hiera_tiny.pt` and `sam2.1_hiera_large.pt`: SAM2.1 baselines used by
  legacy experiments.
- `MedSAM2_latest.pt`: retained MedSAM2 baseline; it was not the final
  champion for gross pathology RGB images.
- `seg_crop_*.pt`: crop-segmenter ablations retained for auditability.
- `detector_*.pt`, `box_refiner.pt`, `grid_clf.pt`, `slic_clf.pt`,
  `mask2former_lite.pt`, `yolo_best.pt`: full-auto/localizer experiments that
  are documented in `experiments/rejected/README.md`.

## Notes For Readers

Do not choose a checkpoint by newest filename. Start from
`configs/model/seg_crop_lab.yaml`; it points to the intended deliverable
checkpoint. See `experiments/best_current/README.md` for why this model is kept
as the best current mask deliverable and why full-auto localization remains the
next research target.
