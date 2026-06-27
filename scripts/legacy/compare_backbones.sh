#!/usr/bin/env bash
# Chờ tải large xong + vòng train hiện tại xong (tránh OOM khi GPU free ít),
# rồi đo zero-shot Dice TINY vs LARGE trên cùng 12 ảnh test.
set -u
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
PY=/home/hvusynh2/conda_envs/medsam2_anno/bin/python
LOG=compare_backbones.log
ST=compare_backbones.status
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
: > "$LOG"; echo "RUNNING" > "$ST"

say "Chờ checkpoint large tải xong..."
for i in $(seq 1 480); do [ -f checkpoints/large_dl.done ] && break; sleep 15; done
if ! grep -q OK checkpoints/large_dl.done 2>/dev/null; then
  say "Tải large THẤT BẠI."; echo "ERROR" > "$ST"; exit 1; fi
sz=$(stat -c%s checkpoints/sam2.1_hiera_large.pt 2>/dev/null || echo 0)
say "Tải xong ($((sz/1024/1024)) MB)."

say "Chờ vòng train/finetune hiện tại xong (tránh tranh GPU)..."
for i in $(seq 1 240); do
  pgrep -f "loop_round.sh" >/dev/null || pgrep -f "finetune_sam2.py" >/dev/null || break
  sleep 15
done

say "Đo zero-shot TINY (baseline, split hiện tại)..."
SAM2_CONFIG=configs/sam2.1_hiera_t512 SAM2_CKPT=checkpoints/sam2.1_hiera_tiny.pt \
  $PY eval_zeroshot.py 2>&1 | grep -vE "UserWarning|warn|FutureWarning|category=" \
  | sed 's/^/    /' | tee -a "$LOG"

say "Đo zero-shot LARGE..."
SAM2_CONFIG=configs/sam2.1_hiera_l SAM2_CKPT=checkpoints/sam2.1_hiera_large.pt \
  $PY eval_zeroshot.py 2>&1 | grep -vE "UserWarning|warn|FutureWarning|category=" \
  | sed 's/^/    /' | tee -a "$LOG"

say "=== XONG. So sánh: ==="
$PY - <<'PY' 2>&1 | sed 's/^/    /' | tee -a "$LOG"
import json,os
for t in ("tiny","large"):
    p=f"results/zeroshot_{t}.json"
    if os.path.isfile(p):
        d=json.load(open(p)); print(f"{t:6s}: Dice={d['zero_shot_dice']} (n_test={d['n_test']})")
PY
echo "DONE" > "$ST"
