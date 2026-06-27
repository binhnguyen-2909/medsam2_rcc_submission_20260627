#!/usr/bin/env bash
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
PY=/home/hvusynh2/conda_envs/medsam2_anno/bin/python
echo "[$(date '+%T')] chờ GPU >= 6000 MiB trống..."
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$FREE" -ge 6000 ]; then echo "[$(date '+%T')] GPU free=${FREE}MiB -> chạy"; break; fi
  sleep 20
done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u scratch_confirm200.py
echo "[$(date '+%T')] DONE confirm200"
