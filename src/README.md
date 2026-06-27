# Source Code

This folder contains the runnable code for the RCC gross-pathology segmentation
package.

## Main Entry Point

Use this for the current deliverable:

```bash
python src/inference/predict_seg_crop.py \
  --config configs/model/seg_crop_lab.yaml \
  --image data/raw/images/CASE.jpg \
  --box "x0,y0,x1,y1" \
  --out experiments/runs/manual_case/predictions/CASE.png \
  --overlay experiments/runs/manual_case/predictions/CASE_overlay.jpg
```

Batch mode expects a CSV with columns:

```csv
image,x0,y0,x1,y1
data/raw/images/case001.jpg,436,666,704,902
```

## Layout

- `inference/`: production/research inference entrypoints. Start here.
- `model/`: reusable model definitions for detector and crop segmenter.
- `data/`: preprocessing and specimen utilities.
- `evaluate/`: legacy evaluation wrappers retained for reproducibility.
- `train/`: legacy training wrappers retained for reproducibility.
- `legacy/`: original scripts copied from the development workspace.
- `external/`: vendored SAM2 code used by legacy baselines.

## Interpretation

The maintained deliverable path is `inference/predict_seg_crop.py` plus
`model/seg_crop_model.py`. The full-auto scripts are retained because they
explain the research trajectory, but their results show localization remains the
bottleneck.
