# MedSAM2 RCC Reproducible Research Package

This folder reorganizes the RCC gross-pathology segmentation work from
`/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2` into a reproducible research
layout. The current deliverable is **semi-auto box-to-mask**:

- user draws tight tumor box(es);
- `SegResNet + RGB/LAB` segments each cropped box;
- masks are merged back to full image resolution.

Current champion from the project memory:

- deliverable: `predict_seg_crop.py`
- checkpoint: `checkpoints/seg_crop_segR_lab.pt`
- independent hand-drawn evaluation: median Dice `0.8834`, mean Dice `0.8682` on `n=50`
- full-auto localizer remains limited: median Dice about `0.635`

## Read This First

For a quick review of the submission, read these files in order:

1. `README.md`: package purpose, setup, and quick inference command.
2. `docs/project_summary.md`: research narrative and final interpretation.
3. `results/README.md`: where the result tables and qualitative outputs live.
4. `experiments/best_current/README.md`: current best semi-auto deliverable,
   best stable full-auto run, and recommended next experiment.
5. `experiments/rejected/README.md`: negative results and directions not to
   repeat.

The key practical path is:

```text
draw or provide tight tumor box(es)
  -> run src/inference/predict_seg_crop.py
  -> inspect output mask and overlay
  -> record the run under experiments/runs/
  -> compare against results/tables/
```

## Directory Layout

```text
configs/                 YAML configuration, no experiment parameter hard-code
data/
  raw/                   original gross pathology images
  ground_truth/          SAM-assisted masks, truth masks, hand-drawn eval masks
  interim/               temporary data generated during preprocessing/training
  processed/             crops, specimen masks, QC output
  manifests/             splits, metadata, annotation logs, Excel-derived tables
src/
  data/                  preprocessing and specimen utilities
  model/                 model definitions
  train/                 training scripts and legacy reproductions
  evaluate/              evaluation scripts and legacy reproductions
  inference/             production/research inference entrypoints
  external/              vendored SAM2 code needed by legacy baselines
checkpoints/             required model weights
models/                  auxiliary legacy localizer weights
experiments/
  _templates/            README template required for every run
  runs/                  one subfolder per run
results/                 curated result tables/figures copied from original work
legacy_artifacts/        original generated artifacts that do not fit clean modules
docs/                    design notes and reproducibility notes
scripts/                 reproducibility utilities and legacy shell runners
notebooks/               optional exploratory notebooks; none required currently
reports/                 optional generated reports; primary docs are in docs/
tests/                   placeholder for future automated tests
```

## Setup

```bash
cd /home/hvusynh2/nguyenduong/medsam2_rcc_submission_20260627
conda env create -f environment.yml
conda activate medsam2-rcc-repro
```

The exact pip lock is in `requirements.txt`. The known-good interpreter used in
the original work was:

```bash
/home/hvusynh2/conda_envs/medsam2_anno/bin/python
```

## Quick Inference

```bash
python src/inference/predict_seg_crop.py \
  --config configs/model/seg_crop_lab.yaml \
  --image data/raw/images/CASE.jpg \
  --box "x0,y0,x1,y1" \
  --out experiments/runs/manual_case/predictions/CASE.png \
  --overlay experiments/runs/manual_case/predictions/CASE_overlay.jpg
```

Batch CSV format:

```csv
image,x0,y0,x1,y1
data/raw/images/case001.jpg,436,666,704,902
data/raw/images/case002.jpg,417,723,732,951
data/raw/images/case002.jpg,900,400,1100,650
```

Run:

```bash
python src/inference/predict_seg_crop.py \
  --config configs/model/seg_crop_lab.yaml \
  --csv boxes.csv \
  --out_dir experiments/runs/run_id/predictions/masks \
  --overlay_dir experiments/runs/run_id/predictions/overlays
```

## Creating a Reproducible Run Folder

Every run must have its own folder with checkpoints, logs, predictions, copied
config, command, environment lock, git status, and a README to fill in:

```bash
python scripts/create_run.py \
  --run-name 20260626_segR_lab_demo \
  --config configs/model/seg_crop_lab.yaml \
  --command 'python src/inference/predict_seg_crop.py --config configs/model/seg_crop_lab.yaml --csv boxes.csv --out_dir experiments/runs/20260626_segR_lab_demo/predictions/masks'
```

Then write notes in:

```text
experiments/runs/20260626_segR_lab_demo/README.md
```

## Rejected Directions

The audit trail for directions that were tried and discarded is maintained in:

```text
experiments/rejected/README.md
```

That file records the script/command family, parameters, theory, result, and
reason each direction was rejected.

The current best run table is maintained in:

```text
experiments/best_current/README.md
```

## Data and Artifact Policy

This local scaffold contains copied raw images, masks, processed artifacts, and
the checkpoints/results needed for the current deliverable, baselines, and legacy
experiment history. The `.gitattributes` file marks binary artifacts for Git LFS;
run `git lfs install` before publishing/cloning through Git.

Legacy material copied from the original workspace:

- `src/legacy/`: original top-level Python scripts.
- `scripts/legacy/`: original shell runners.
- `docs/original/`: original markdown/status docs.
- `results/original/` and `results/original_195/`: original result tables, logs, and figures.
- `legacy_artifacts/`: original run folders, examples, training assets, yolo dataset, and loose root files.
- `data/deliverable_dataset/`: packaged mask dataset from the original deliverable.
- `data/raw/metadata/`: original spreadsheet metadata, including `RCC 20241212.xlsx`.
- `data/manifests/labels*`: original annotation/split/queue metadata from label folders.
- `legacy_artifacts/scratch/`: original scratch outputs kept for auditability.

Use checksums before publishing:

```bash
python scripts/export_checksums.py --include-large-data --out data/manifests/artifact_checksums.sha256
python scripts/validate_artifacts.py --manifest data/manifests/artifact_checksums.sha256
```

## Key Caveats

- The data are gross pathology kidney RCC photos, not radiology.
- Excel tumor size is useful for sanity checks, not for locating the tumor.
- `px/cm` from ruler detection was unreliable and should not drive QC.
- Full-auto localization is the bottleneck; the reliable deliverable is semi-auto
  tight box(es) to tumor mask.

## Recommended Next Direction

If continuing this work, prioritize localization rather than changing the
box-to-mask segmenter. The best current semi-auto segmenter is already strong
when boxes are tight. The next full-auto experiment to try is documented in
`experiments/best_current/README.md`: run the `component_strict comp4` variant
on the full dataset and compare its stability against
`full_auto_specimen_strict_20260626`.
