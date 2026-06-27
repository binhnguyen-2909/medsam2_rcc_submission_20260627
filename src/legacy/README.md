# Legacy Source

This directory contains original Python scripts copied from the development
workspace. They are retained so historical results and rejected directions can
be audited.

## Use The Maintained Path First

For the current deliverable, use:

```text
src/inference/predict_seg_crop.py
src/model/seg_crop_model.py
configs/model/seg_crop_lab.yaml
checkpoints/seg_crop_segR_lab.pt
```

## When To Read Legacy Scripts

Read this directory only when tracing a result mentioned in:

- `experiments/best_current/README.md`
- `experiments/rejected/README.md`
- `results/original/`

Many scripts encode earlier assumptions or absolute paths from the original
workspace. They are preserved for provenance, not recommended as new entrypoints.
