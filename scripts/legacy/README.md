# Legacy Scripts

These shell scripts are copied from the original development workspace. They
preserve the commands used during exploration and ablation.

## How To Use

Do not start here for the final deliverable. Use:

```text
src/inference/predict_seg_crop.py
configs/model/seg_crop_lab.yaml
checkpoints/seg_crop_segR_lab.pt
```

The legacy scripts may contain absolute paths to the original workspace. Treat
them as audit trail for how historical results were produced, not as polished
portable commands.

## Where Results Are Explained

The outcomes of these directions are summarized in
`experiments/rejected/README.md` and `experiments/best_current/README.md`.
