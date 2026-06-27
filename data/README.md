# Data Layout

```text
raw/images/                         1392 gross pathology RCC images
raw/metadata/                       original Excel spreadsheet and raw metadata
ground_truth/sam_assisted_masks/    masks generated with SAM-assisted annotation
ground_truth/sam_assisted_prompts/  saved boxes/points used during annotation
ground_truth/truth_masks/           curated truth masks used for later training
ground_truth/handdraw_eval_masks/   independent hand-drawn evaluation masks
processed/crop/                     specimen crops from preprocessing
processed/specimen_masks/           specimen masks, not tumor masks
processed/qc/                       preprocessing QC images
manifests/                          splits, metadata, logs, result manifests
manifests/labels*/                  original annotation state files
deliverable_dataset/                 original packaged deliverable mask dataset
interim/yolo_ds/                     generated YOLO dataset used by localizer experiments
```

Important distinctions:

- `raw/images` is immutable raw input.
- `processed/specimen_masks` is the kidney/specimen region, not the tumor.
- Tumor masks live under `ground_truth`.
- `handdraw_eval_masks` is the most important set for unbiased evaluation.

The large folders have been copied into this scaffold so local reruns do not
depend on the original workspace. When publishing the project, keep these paths
unchanged and track binary artifacts with Git LFS or DVC. Validate transferred
artifacts with `data/manifests/artifact_checksums.sha256`.
