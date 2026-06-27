# Scripts

This folder contains reproducibility utilities and legacy shell runners.

## Current Utilities

- `create_run.py`: creates a documented folder under `experiments/runs/` with
  copied config, environment files, command, git status, and README template.
- `export_checksums.py`: writes SHA256 checksums for reproducibility-critical
  files.
- `validate_artifacts.py`: validates files against a checksum manifest.

Example:

```bash
python scripts/create_run.py \
  --run-name smoke_seg_crop_lab \
  --config configs/model/seg_crop_lab.yaml \
  --command 'python src/inference/predict_seg_crop.py --config configs/model/seg_crop_lab.yaml --image data/raw/images/CASE.jpg --box "x0,y0,x1,y1" --out experiments/runs/smoke_seg_crop_lab/predictions/CASE.png'
```

## Legacy Runners

`scripts/legacy/` contains the original shell commands used during exploration
and ablation. They are kept for auditability. Prefer the top-level Python
utilities and documented configs for new reproducible runs.
