#!/bin/bash
# MAE: chờ adv runner xong -> pretrain trên 1128 ảnh -> finetune (MAE) + finetune (scratch đối chứng).
# Detached: setsid nohup bash run_mae.sh > results/mae_all.log 2>&1 &
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
PY=/home/hvusynh2/conda_envs/medsam2_anno/bin/python
MINMB=6000
gate() {
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$MINMB" ]; then echo "[gate] GPU free=${free}MB -> chạy"; break; fi
    echo "[gate] GPU free=${free}MB < ${MINMB}, chờ 60s..."; sleep 60
  done
}
# chờ adv runner xong để khỏi tranh GPU
echo "[mae] chờ adv runner (results/seg_adv_all.done)..."
while [ ! -f results/seg_adv_all.done ]; do sleep 120; done
echo "[$(date +%H:%M:%S)] adv xong -> bắt đầu MAE"

echo "==== PRETRAIN MAE (1128 ảnh) ===="; gate
$PY -u mae_seg.py --pretrain --epochs 40 --batch 8 2>&1
echo "==== FINETUNE (MAE encoder) ===="; gate
$PY -u mae_seg.py --finetune --tag seg_mae --epochs 60 --batch 8 2>&1
echo "==== FINETUNE (scratch đối chứng) ===="; gate
$PY -u mae_seg.py --finetune --scratch --tag seg_unet_scratch --epochs 60 --batch 8 2>&1
echo "TẤT CẢ XONG" > results/mae_all.done
echo "[$(date +%H:%M:%S)] ===== XONG MAE ====="
