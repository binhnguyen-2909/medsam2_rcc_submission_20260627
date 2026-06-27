# Reproducibility Checklist

Before a result is considered reproducible, the corresponding run folder must
contain:

- exact execution command in `logs/command.txt` and `README.md`;
- copied YAML config under `configs/`;
- copied `requirements.txt` / `environment.yml` under `environment/`;
- checkpoint files or checksum references under `checkpoints/`;
- stdout/stderr logs under `logs/`;
- predictions under `predictions/`;
- metrics table under `results/` or inside the run folder;
- analysis explaining why the method was tried, why it succeeded or failed, and
  what the next experiment should be.

Recommended command:

```bash
python scripts/create_run.py --run-name RUN_ID --config CONFIG.yaml --command 'COMMAND'
```

Then run the command and fill in `experiments/runs/RUN_ID/README.md`.
