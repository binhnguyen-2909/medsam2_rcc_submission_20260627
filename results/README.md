# Results

This folder contains curated evaluation tables, figures, logs, and original
result artifacts retained for auditability.

## Primary Tables

The main summary files are in `results/tables/`:

- `seg_crop_segR_lab.json`: champion semi-auto box-to-mask summary. On the
  independent hand-drawn set (`n=50`), the tight-box ceiling is median Dice
  `0.8834` and mean Dice `0.8682`. The same file also records full-auto
  detector performance for comparison.
- `loc_detector.json`: full-auto detector localization summary on the same
  evaluation split (`n=50`), with median Dice `0.6347` and mean Dice `0.5897`.
- `handdraw_eval.csv`: per-case hand-drawn evaluation table used for the
  independent evaluation.
- `best_full_auto_detector_recall_eval.csv`: per-case full-auto detector
  evaluation table for the best detector-recall baseline.

## Original Artifacts

- `results/original/`: original tables, logs, montages, and figures copied from
  the development workspace. These files preserve experiment history and
  negative/legacy baselines.
- `results/original/full_auto_component_strict_comp2_20260628/`: full-data
  conservative component-strict run on all `1393` images. It contains `1393`
  masks, `1393` overlay thumbnails, `summary.csv`, and `boxes.csv`.
- `results/original_195/`: selected individual qualitative outputs and an
  inference grid from the original 195-case result set.
- `results/figures/`: reserved for curated publication/report figures.

## Interpretation Notes

The reliable deliverable in this repository is the semi-auto tight box to tumor
mask workflow implemented by `src/inference/predict_seg_crop.py` with
`checkpoints/seg_crop_segR_lab.pt`.

Full-auto localization results are included for transparency, but they are not
the champion deliverable because localization remains the bottleneck. See the
top-level `README.md` and `docs/reproducibility.md` for setup, inference, and
reproducibility instructions.
