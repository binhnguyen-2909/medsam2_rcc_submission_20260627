#!/bin/bash
# SETUP & CHẠY TẤT CẢ localizer (ý tưởng #1-#6) -> segmenter champion -> eval 50 vẽ tay, lưu JSON.
# Detached: setsid nohup bash run_localize.sh > results/localize_all.log 2>&1 &
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
PY=/home/hvusynh2/conda_envs/medsam2_anno/bin/python
PIP=/home/hvusynh2/conda_envs/medsam2_anno/bin/pip
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MINMB=6000
gate() { while true; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$MINMB" ]; then echo "[gate] GPU free=${free}MB -> chạy"; break; fi
    echo "[gate] GPU ${free}MB < ${MINMB}, chờ 60s..."; sleep 60; done; }
ev() { echo "==== EVAL $1 ===="; gate; $PY -u localize_eval.py --method "$1" 2>&1; }

echo "[$(date +%H:%M:%S)] ===== BẮT ĐẦU LOCALIZE PIPELINE ====="

# 0) baseline + các method KHÔNG cần train trước
ev detector
ev iter
ev centerpoint

# #2 BOX REFINER
echo "==== TRAIN box_refiner ===="; gate; $PY -u box_refiner.py --epochs 80 2>&1
ev refiner

# #1 YOLOv11 (cài ultralytics nếu thiếu)
echo "==== SETUP ultralytics ===="
$PY -c "import ultralytics" 2>/dev/null || timeout 400 $PIP install ultralytics 2>&1 | tail -3
$PY -c "import ultralytics; print('ultralytics', ultralytics.__version__)" 2>&1 | tail -1
echo "==== BUILD+TRAIN yolo ===="; gate; $PY -u train_yolo.py --build --train --model yolo11s.pt --epochs 80 2>&1
[ -f checkpoints/yolo_best.pt ] && ev yolo || echo "[yolo] không có best.pt -> bỏ eval"

# #5 SLIC superpixel
echo "==== TRAIN slic_clf ===="; gate; $PY -u slic_clf.py --epochs 60 2>&1
ev slic

# #6 grid classifier
echo "==== TRAIN grid_clf ===="; gate; $PY -u grid_clf.py --epochs 30 2>&1
ev grid

# #3 Grounding DINO (best-effort, không chặn các cái khác)
echo "==== (tùy chọn) Grounding DINO ===="
$PY -c "import groundingdino" 2>/dev/null && ev gdino || echo "[gdino] chưa cài groundingdino -> bỏ qua (cần repo+weights offline)"

echo "TẤT CẢ XONG" > results/localize_all.done
echo "[$(date +%H:%M:%S)] ===== XONG LOCALIZE PIPELINE ====="
echo "=== TỔNG HỢP ==="; $PY -c "
import json,glob
rows=[json.load(open(f)) for f in sorted(glob.glob('results/loc_*.json'))]
print(f\"{'method':14}{'median':>8}{'mean':>8}{'1u':>8}{'>1u':>8}\")
for d in sorted(rows,key=lambda x:-x['median']):
    print(f\"{d['method']:14}{d['median']:8.4f}{d['mean']:8.4f}{d['1u']:8.3f}{d['>1u']:8.3f}\")
print('SO: full-auto detector~0.635 | ceiling box-GT 0.883')
" 2>&1
