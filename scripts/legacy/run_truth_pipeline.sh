#!/bin/bash
# Train detector box-only center-sampling trên mask THẬT (labels_truth) -> eval handdraw.
# Detached (setsid) để sống sót teardown phiên. Gate chờ GPU >=6GB (A100 dùng chung).
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
PY=/home/hvusynh2/conda_envs/medsam2_anno/bin/python
LOG=results/detector_truth_train.log
echo "[gate] bắt đầu chờ GPU $(date +%H:%M)" > results/detector_truth_gate.log

# chờ tối đa ~6h — ngưỡng 2800MiB (user yêu cầu chạy ở ~3GB; chỉ precompute SAM@1024 ~36 ảnh cần RAM)
for i in $(seq 1 720); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -ge 2800 ]; then
    echo "[gate] GPU free ${free}MiB >= 2800 -> train $(date +%H:%M)" >> results/detector_truth_gate.log
    break
  fi
  sleep 30
done

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u train_detector.py \
  --epochs 45 --batch 4 --mask_warmup 999 --min_pos 3 --grid 64 \
  --eval_every 1 --ckpt_out checkpoints/detector_truth.pt > "$LOG" 2>&1
echo "[gate] train xong $(date +%H:%M)" >> results/detector_truth_gate.log

if [ -f checkpoints/detector_truth.pt ]; then
  echo "[chain] eval handdraw $(date +%H:%M)" >> results/detector_truth_gate.log
  DET_CKPT=checkpoints/detector_truth.pt "$PY" -u eval_handdraw.py > results/eval_handdraw_truth.log 2>&1
  cp results/handdraw_eval.csv results/handdraw_eval_truth.csv
  cp results/handdraw_eval_recall.csv results/handdraw_eval.csv
  echo "[chain] XONG $(date +%H:%M); default csv = grid64" >> results/detector_truth_gate.log
else
  echo "[chain] KHÔNG có ckpt - train lỗi" >> results/detector_truth_gate.log
fi
