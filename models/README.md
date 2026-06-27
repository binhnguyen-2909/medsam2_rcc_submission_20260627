# Models

This folder contains auxiliary detector/localizer weights from earlier
experiments.

## Files

- `cellbox.pt` and `cellbox_ensemble.pt`: grid/cell box proposal models from
  rejected auto-box experiments.
- `yolo11s.pt` and `yolo26n.pt`: YOLO localizer weights retained for legacy
  comparison.

## Main Checkpoint Location

The current deliverable checkpoint is not in this folder. Use:

```text
checkpoints/seg_crop_segR_lab.pt
```

with:

```text
configs/model/seg_crop_lab.yaml
src/inference/predict_seg_crop.py
```

The weights in `models/` are retained so historical localizer experiments can be
audited, but they are not the recommended submission path.
