#!/bin/bash
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
PY=/home/hvusynh2/conda_envs/medsam2_anno/bin/python
G=results/seg_sam_gate.log
echo "[gate] chờ GPU $(date +%H:%M)" > "$G"
for i in $(seq 1 720); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -ge 2800 ]; then echo "[gate] GPU ${free}MiB -> chạy $(date +%H:%M)" >> "$G"; break; fi
  sleep 30
done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -u seg_sam.py --epochs 80 --batch 16 > results/seg_sam.log 2>&1
echo "[gate] XONG $(date +%H:%M)" >> "$G"
