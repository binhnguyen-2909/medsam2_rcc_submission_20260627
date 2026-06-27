#!/bin/bash
# 3 localizer mới: #A AMG+classifier, #B anomaly-AE, #3 size-constraint. Lưu loc_<m>.json.
# Detached: setsid nohup bash run_localize2.sh > results/localize2.log 2>&1 &
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
PY=/home/hvusynh2/conda_envs/medsam2_anno/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ev(){ echo "[$(date +%H:%M:%S)] EVAL $1"; $PY -u localize_eval.py --method "$1" > results/loc_${1}.log 2>&1; tail -3 results/loc_${1}.log | grep -i median; echo "[$(date +%H:%M:%S)] $1 xong"; }

echo "[$(date +%H:%M:%S)] ===== TRAIN amg-clf + anomaly-ae (song song) ====="
$PY -u amg_classify.py --epochs 25 > results/train_amgclf.log 2>&1 & PA=$!
$PY -u anomaly_ae.py   --epochs 40 > results/train_ae.log     2>&1 & PB=$!
wait $PA $PB
echo "[$(date +%H:%M:%S)] train xong: $(ls checkpoints/amg_clf.pt checkpoints/anomaly_ae.pt 2>/dev/null)"

ev size       # nhanh (lọc box detector theo Excel)
ev anomaly    # quét AE
ev amg        # chậm nhất (grid-point SAM)

echo "TẤT CẢ XONG" > results/localize2.done
echo "[$(date +%H:%M:%S)] ===== XONG 3 HƯỚNG MỚI ====="
$PY -c "
import json,glob
rows=[json.load(open(f)) for f in sorted(glob.glob('results/loc_*.json'))]
print(f\"{'method':14}{'median':>9}{'mean':>9}{'1u':>8}{'>1u':>8}\")
for d in sorted(rows,key=lambda x:-x['median']):
    print(f\"{d['method']:14}{d['median']:9.4f}{d['mean']:9.4f}{d['1u']:8.3f}{d['>1u']:8.3f}\")
print('SO: detector 0.635 | ceiling 0.883')
"
