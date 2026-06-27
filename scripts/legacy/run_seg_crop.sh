#!/bin/bash
# Chạy tuần tự các ablation seg-crop (two-stage crop + segmenter thuần).
# Gate GPU dùng chung: chờ memory.free >= MINMB rồi mới chạy mỗi config.
# Detached: setsid nohup bash run_seg_crop.sh > results/seg_crop_all.log 2>&1 &
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
PY=/home/hvusynh2/conda_envs/medsam2_anno/bin/python
MINMB=6000
EP=60

gate() {
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$MINMB" ]; then echo "[gate] GPU free=${free}MB >= ${MINMB} -> chạy"; break; fi
    echo "[gate] GPU free=${free}MB < ${MINMB}, chờ 60s..."; sleep 60
  done
}

run() {  # $1=args $2=tag
  echo "======================================================================"
  echo "[$(date +%H:%M:%S)] BẮT ĐẦU $2"
  gate
  $PY -u seg_crop.py $1 --tag "$2" --epochs $EP --batch 6 2>&1
  echo "[$(date +%H:%M:%S)] XONG $2"
}

# 1) baseline pipeline mới: RGB thuần
run "--arch segresnet --channels rgb --loss dicebce"          "segR_rgb"
# 2) + kênh màu LAB (6ch)
run "--arch segresnet --channels lab --loss dicebce"          "segR_lab"
# 3) + kênh texture Gabor (7ch)
run "--arch segresnet --channels lab_tex --loss dicebce"      "segR_labtex"
# 4) + Boundary loss (kết hợp kênh tốt nhất)
run "--arch segresnet --channels lab_tex --loss diceboundary" "segR_labtex_bd"
# 5) đổi backbone SwinUNETR (cấu hình tốt nhất)
run "--arch swinunetr  --channels lab_tex --loss diceboundary" "swin_labtex_bd"

echo "TẤT CẢ XONG" > results/seg_crop_all.done
echo "[$(date +%H:%M:%S)] ===== TẤT CẢ ABLATION XONG ====="
