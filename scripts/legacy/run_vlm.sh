#!/bin/bash
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
PY=/home/hvusynh2/conda_envs/medsam2_anno/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1)
  [ "$free" -ge 4500 ] && { echo "[gate] GPU ${free}MB -> chạy"; break; }
  echo "[gate $(date +%H:%M)] GPU ${free}MB <4500, chờ"; sleep 60
done
$PY -u vlm_eval.py 2>&1
echo DONE > results/vlm.done
