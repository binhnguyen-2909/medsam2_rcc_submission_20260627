# Legacy Artifacts

This folder preserves files copied from the original development workspace that
do not fit cleanly into `src/`, `data/`, `results/`, or `experiments/`.

## Contents

- `root_files/`: loose logs, status files, CSVs, and scratch outputs from the
  original root workspace.
- `runs/`: original detector/output run folders retained for traceability.
- `scratch/`: scratch masks, overlays, and QC images used during exploration.
- `training/`: original training package fragments and assets.
- `efficient_track_anything/`: copied external code used by legacy experiments.
- `examples/`: legacy example scripts from upstream or earlier experiments.

## How To Interpret

These files are not the recommended entrypoint for reproducing the final
deliverable. Start with the top-level `README.md`, `src/README.md`, and
`results/README.md`. Use this folder only when auditing historical decisions or
checking provenance for a legacy result mentioned in `experiments/rejected/`.

New work should not write outputs here. Put new reproducible runs under
`experiments/runs/`.
