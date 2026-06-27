#!/usr/bin/env bash
# Train seg_sam trên đặc trưng SAM2.1 LARGE encoder (đông cứng) -> eval 50 vẽ tay.
# Chờ large tải xong + GPU đủ ~13GB rồi chạy. Detached setsid, sống sót tắt phiên.
set -u
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
PY=/home/hvusynh2/conda_envs/medsam2_anno/bin/python
G=results/seg_sam_large_gate.log
say(){ echo "[gate] $* $(date '+%H:%M')" | tee -a "$G"; }
: > "$G"

say "chờ large tải xong"
for i in $(seq 1 480); do [ -f checkpoints/large_dl.done ] && break; sleep 15; done
grep -q OK checkpoints/large_dl.done 2>/dev/null || { say "TẢI LARGE LỖI"; exit 1; }
say "large OK $(stat -c%s checkpoints/sam2.1_hiera_large.pt | awk '{print int($1/1024/1024)}')MB"

say "chờ GPU >=13000MiB trống"
for i in $(seq 1 2880); do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "${f:-0}" -ge 13000 ] && { say "GPU ${f}MiB -> chạy"; break; }
  sleep 15
done

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SAM2_CONFIG=configs/sam2.1_hiera_l \
SAM2_CKPT=checkpoints/sam2.1_hiera_large.pt \
SAM2_RES=1024 \
  $PY -u seg_sam.py --epochs 80 --batch 16 --ckpt_out checkpoints/seg_sam_large.pt \
  > results/seg_sam_large.log 2>&1
say "XONG"
