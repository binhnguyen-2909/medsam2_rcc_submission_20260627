# Run: smoke_seg_crop_lab

## Execution Command

```bash
CUDA_VISIBLE_DEVICES= python src/inference/predict_seg_crop.py --config configs/model/seg_crop_lab.yaml --image "data/raw/images/SS22-02205^2022_01_12_09_37_05^^.jpg" --box "765,489,2065,1429" --out experiments/runs/smoke_seg_crop_lab/predictions/SS22-02205_mask.png --overlay experiments/runs/smoke_seg_crop_lab/predictions/SS22-02205_overlay.jpg
```

## Parameters

- Config file: `configs/model/seg_crop_lab.yaml`
- Main hyperparameters:
  - Fill in or paste the relevant YAML block here.

## Theoretical Basis

Which paper, model family, or methodological assumption motivated this setup?

## Results

- Metrics:
- Correctness ratio / Dice / IoU / HD95:
- Output files:

## Analysis

Why was this initial direction chosen?

If the result was unsatisfactory, what is the likely cause?

Why should this method be kept, modified, or discarded?

What is the logic for the next experiment?

## Reproducibility Record

- Created at: `2026-06-26T18:20:10`
- Git commit: `unavailable: Command '['git', 'rev-parse', 'HEAD']' returned non-zero exit status 128.`
- Working tree status: see `logs/git_status.txt`
- Environment lock: `environment/requirements.txt`
- Config copy: `configs/`
- Checkpoints: `checkpoints/`
- Predictions: `predictions/`
- Logs: `logs/`
