#!/bin/bash
# CHẠY SONG SONG các localizer độc lập (tận dụng GPU trống). detector/iter/centerpoint đã xong.
# Phase 1 (//): train refiner + slic + grid đồng thời (nhẹ; slic/grid nặng CPU -> overlap tốt).
# Phase 2: YOLO (nặng, chạy riêng). Phase 3: eval tất cả. Tự lưu loc_<method>.json.
# Detached: setsid nohup bash run_localize_par.sh > results/localize_par.log 2>&1 &
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
PY=/home/hvusynh2/conda_envs/medsam2_anno/bin/python
PIP=/home/hvusynh2/conda_envs/medsam2_anno/bin/pip
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ev() { echo "[$(date +%H:%M:%S)] EVAL $1"; $PY -u localize_eval.py --method "$1" > results/loc_${1}.log 2>&1; echo "[$(date +%H:%M:%S)] EVAL $1 xong"; }

echo "[$(date +%H:%M:%S)] ===== PHASE 1: TRAIN SONG SONG (refiner + slic + grid) ====="
$PY -u box_refiner.py --epochs 80 > results/train_refiner.log 2>&1 & PR=$!
$PY -u slic_clf.py    --epochs 60 > results/train_slic.log    2>&1 & PS=$!
$PY -u grid_clf.py    --epochs 30 > results/train_grid.log    2>&1 & PG=$!
echo "PIDs refiner=$PR slic=$PS grid=$PG — chờ cả 3..."
wait $PR $PS $PG
echo "[$(date +%H:%M:%S)] PHASE 1 xong. ckpt:"; ls -1 checkpoints/box_refiner.pt checkpoints/slic_clf.pt checkpoints/grid_clf.pt 2>/dev/null

echo "[$(date +%H:%M:%S)] ===== PHASE 2: YOLOv11 ====="
$PY -c "import ultralytics" 2>/dev/null || timeout 400 $PIP install ultralytics > results/pip_ultra.log 2>&1
$PY -c "import ultralytics; print('ultralytics', ultralytics.__version__)" 2>&1 | tail -1
$PY -u train_yolo.py --build --train --model yolo11s.pt --epochs 80 > results/train_yolo.log 2>&1
echo "[$(date +%H:%M:%S)] YOLO xong. best: $(ls checkpoints/yolo_best.pt 2>/dev/null || echo 'KHÔNG có')"

echo "[$(date +%H:%M:%S)] ===== PHASE 3: EVAL (song song 2 luồng) ====="
ev refiner & ev slic & wait
ev grid &
[ -f checkpoints/yolo_best.pt ] && ev yolo
wait
# gdino best-effort
$PY -c "import groundingdino" 2>/dev/null && ev gdino || echo "[gdino] bỏ qua (chưa cài)"

echo "TẤT CẢ XONG" > results/localize_all.done
echo "[$(date +%H:%M:%S)] ===== XONG ====="
echo "=== BẢNG TỔNG HỢP ==="; $PY -c "
import json,glob
rows=[json.load(open(f)) for f in sorted(glob.glob('results/loc_*.json'))]
print(f\"{'method':14}{'median':>9}{'mean':>9}{'1u':>8}{'>1u':>8}\")
for d in sorted(rows,key=lambda x:-x['median']):
    print(f\"{d['method']:14}{d['median']:9.4f}{d['mean']:9.4f}{d['1u']:8.3f}{d['>1u']:8.3f}\")
print('SO: full-auto detector 0.635 | ceiling box-GT 0.883 | nhãn SAM cũ 0.554')
"
