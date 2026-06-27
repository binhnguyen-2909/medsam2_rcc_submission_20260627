#!/bin/bash
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
PY=/home/hvusynh2/conda_envs/medsam2_anno/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MINMB=7000
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -ge "$MINMB" ]; then echo "[gate] GPU free=${free}MB -> chạy M2F"; break; fi
  echo "[gate $(date +%H:%M)] GPU ${free}MB < ${MINMB}, chờ..."; sleep 90
done
$PY -u mask2former_lite.py --epochs 80 --batch 4 2>&1
echo "M2F DONE" > results/m2f.done
