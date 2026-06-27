#!/usr/bin/env bash
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
PY=/home/hvusynh2/conda_envs/medsam2_anno/bin/python
NEED=5000
echo "[$(date '+%F %T')] chờ GPU >= ${NEED} MiB (ổn định 2 lần)..."
ok=0
while true; do
  F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$F" -ge "$NEED" ]; then ok=$((ok+1)); echo "[$(date '+%T')] free=${F} (lần $ok/2)"; else ok=0; fi
  [ "$ok" -ge 2 ] && { echo "[$(date '+%T')] đủ GPU -> CHẠY pipeline"; break; }
  sleep 20
done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u e2e_pipeline.py
echo "[$(date '+%F %T')] DONE e2e_pipeline (exit $?)"
