# Experiments

This folder records what has been run, what currently works best, and what
should be tried next.

## Structure

- `_templates/README.md`: template used by `scripts/create_run.py` for new run
  folders.
- `runs/`: concrete reproducible runs. Each run should contain copied config,
  command, environment lock, git status, predictions, logs, and a README.
- `best_current/README.md`: current best run table and next recommended
  direction.
- `rejected/README.md`: experiment directions that were tested and rejected,
  with reasons and result references.

## Create A New Run

```bash
python scripts/create_run.py \
  --run-name 20260627_my_run \
  --config configs/model/seg_crop_lab.yaml \
  --command 'python src/inference/predict_seg_crop.py --config configs/model/seg_crop_lab.yaml --csv boxes.csv --out_dir experiments/runs/20260627_my_run/predictions/masks'
```

Then run the recorded command and fill in the generated run README with the
observed result, failure mode, and decision.

## Current Direction

For the deliverable, keep using the semi-auto crop segmenter
`seg_crop_segR_lab`. For full-auto research, the next direction is improving
localization, especially the `component_strict comp4` variant described in
`best_current/README.md`.
