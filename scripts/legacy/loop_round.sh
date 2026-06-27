#!/usr/bin/env bash
# =====================================================================
# MỘT VÒNG human-in-the-loop: lấy TẤT CẢ nhãn người hiện có ->
#   1. split (test 12 ảnh ĐÓNG BĂNG, nhãn mới vào train)
#   2. fine-tune SAM2.1 + eval trên test cố định
#   3. cập nhật đường cong Dice-theo-số-nhãn
#
# Chạy MỖI KHI bạn vừa gán thêm một mẻ ảnh (qua app QUEUE):
#   cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
#   setsid nohup bash loop_round.sh > /dev/null 2>&1 &      # chịu tắt máy
#   tail -f loop_round.log   |   cat loop_round.status
# =====================================================================
set -u
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
PY=/home/hvusynh2/conda_envs/medsam2_anno/bin/python
LOG=loop_round.log
ST=loop_round.status
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

: > "$LOG"; echo "RUNNING" > "$ST"
say "=== loop_round bắt đầu (pid $$) ==="

say "[1/3] split (test đóng băng)"
$PY split_dataset.py 2>&1 | sed 's/^/    /' | tee -a "$LOG" \
  || { echo "ERROR" > "$ST"; say "split lỗi"; exit 1; }

say "[2/3] fine-tune + eval (test cố định)"
$PY finetune_sam2.py --epochs 60 --eval_every 5 2>&1 \
  | grep -vE "UserWarning|warnings.warn|FutureWarning|torch.cuda.amp|category=" \
  | sed 's/^/    /' | tee -a "$LOG" \
  || { echo "ERROR" > "$ST"; say "finetune lỗi"; exit 1; }

say "[3/3] cập nhật đường cong"
$PY update_curve.py 2>&1 | sed 's/^/    /' | tee -a "$LOG" \
  || { echo "ERROR" > "$ST"; say "update_curve lỗi"; exit 1; }

say "=== VÒNG HOÀN TẤT. Xem results/loop_curve.png ==="
echo "DONE" > "$ST"
