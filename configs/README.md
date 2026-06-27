# Configs

This folder holds YAML files that make the reproducible commands explicit.
Use these configs instead of hard-coding paths or model parameters in scripts.

## Main Configs

- `model/seg_crop_lab.yaml`: champion semi-auto box-to-mask configuration.
  This is the default config for `src/inference/predict_seg_crop.py`.
- `model/full_auto_detector.yaml`: full-auto detector plus segmenter baseline.
  This is included for comparison; localization remains the bottleneck.
- `model/sam2_tiny_baseline.yaml`: SAM2.1 tiny baseline retained for legacy
  comparison.
- `data/rcc_gross.yaml`: dataset path conventions for RCC gross pathology
  images and annotations.
- `evaluate/handdraw50.yaml`: independent hand-drawn evaluation split.
- `train/seg_crop.yaml`: training configuration for the crop segmenter family.

## How To Use

For the current deliverable:

```bash
python src/inference/predict_seg_crop.py \
  --config configs/model/seg_crop_lab.yaml \
  --image data/raw/images/CASE.jpg \
  --box "x0,y0,x1,y1" \
  --out experiments/runs/manual_case/predictions/CASE.png \
  --overlay experiments/runs/manual_case/predictions/CASE_overlay.jpg
```

When adding a new experiment, copy the config into the run folder with
`scripts/create_run.py` so the command remains reproducible.
