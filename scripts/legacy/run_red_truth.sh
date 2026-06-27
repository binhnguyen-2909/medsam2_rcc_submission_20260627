#!/bin/bash
# Fine-tune "đỏ" trên mask thật -> eval handdraw. Detached, gate GPU>=2800 (chạy ở ~3GB).
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
PY=/home/hvusynh2/conda_envs/medsam2_anno/bin/python
G=results/red_truth_gate.log
echo "[gate] chờ GPU $(date +%H:%M)" > "$G"
for i in $(seq 1 720); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -ge 2800 ]; then echo "[gate] GPU ${free}MiB -> chạy $(date +%H:%M)" >> "$G"; break; fi
  sleep 30
done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# baseline ZS cùng harness (xác nhận 0.666/0.857)
echo "[step] eval ZS baseline $(date +%H:%M)" >> "$G"
RED_CKPT="" "$PY" -u eval_red_handdraw.py > results/eval_red_ZS.log 2>&1
# train đỏ
echo "[step] train đỏ $(date +%H:%M)" >> "$G"
"$PY" -u finetune_red_truth.py --epochs 40 --batch 12 --precompute_batch 4 --jitter 0.25 \
  --ckpt_out checkpoints/sam2.1_rcc_red_truth.pt > results/finetune_red_truth.log 2>&1
# eval FT
if [ -f checkpoints/sam2.1_rcc_red_truth.pt ]; then
  echo "[step] eval FT $(date +%H:%M)" >> "$G"
  RED_CKPT=checkpoints/sam2.1_rcc_red_truth.pt "$PY" -u eval_red_handdraw.py > results/eval_red_FT.log 2>&1
  echo "[gate] XONG $(date +%H:%M)" >> "$G"
else
  echo "[gate] LỖI: không có ckpt FT" >> "$G"
fi
