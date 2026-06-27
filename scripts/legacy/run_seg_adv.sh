#!/bin/bash
# Chạy tuần tự 3 config chống-spill: adversarial, clDice, adv+clDice (nền segR_lab).
# Detached: setsid nohup bash run_seg_adv.sh > results/seg_adv_all.log 2>&1 &
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
PY=/home/hvusynh2/conda_envs/medsam2_anno/bin/python
MINMB=6000; EP=60
gate() {
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$MINMB" ]; then echo "[gate] GPU free=${free}MB >= ${MINMB} -> chạy"; break; fi
    echo "[gate] GPU free=${free}MB < ${MINMB}, chờ 60s..."; sleep 60
  done
}
run() { echo "===================================================================="; echo "[$(date +%H:%M:%S)] BẮT ĐẦU $2"; gate; $PY -u seg_crop_adv.py $1 --tag "$2" --epochs $EP 2>&1; echo "[$(date +%H:%M:%S)] XONG $2"; }

run "--adv"            "segR_lab_adv"
run "--cldice"         "segR_lab_cldice"
run "--adv --cldice"   "segR_lab_adv_cldice"
echo "TẤT CẢ XONG" > results/seg_adv_all.done
echo "[$(date +%H:%M:%S)] ===== XONG ADV/clDICE ====="
