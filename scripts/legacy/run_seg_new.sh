#!/bin/bash
# 3 hướng bổ sung: chờ MAE xong -> (1) two-stream FFT, (2) body-edge, (3) hậu xử lý refine.
# Detached: setsid nohup bash run_seg_new.sh > results/seg_new_all.log 2>&1 &
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
PY=/home/hvusynh2/conda_envs/medsam2_anno/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MINMB=6000; EP=60
gate() {
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$MINMB" ]; then echo "[gate] GPU free=${free}MB -> chạy"; break; fi
    echo "[gate] GPU free=${free}MB < ${MINMB}, chờ 60s..."; sleep 60
  done
}
echo "[new] chờ MAE xong (results/mae_all.done)..."
while [ ! -f results/mae_all.done ]; do sleep 120; done
echo "[$(date +%H:%M:%S)] MAE xong -> bắt đầu 3 hướng bổ sung"

echo "==== (1) TWO-STREAM FFT ===="; gate
$PY -u seg_crop_fft.py --channels lab --tag fft_lab --epochs $EP --batch 4 2>&1
echo "==== (2) BODY-EDGE DECOUPLING ===="; gate
$PY -u seg_crop_be.py --channels lab --tag be_lab --epochs $EP --batch 6 2>&1
echo "==== (3) HẬU XỬ LÝ refine (Snakes/guided trên champion segR_lab) ===="; gate
$PY -u refine_postproc.py 2>&1
echo "TẤT CẢ XONG" > results/seg_new_all.done
echo "[$(date +%H:%M:%S)] ===== XONG 3 HƯỚNG BỔ SUNG ====="
