#!/usr/bin/env bash
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
while true; do F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); [ "$F" -ge 6000 ] && { echo "GPU $F -> run"; break; }; sleep 20; done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/hvusynh2/conda_envs/medsam2_anno/bin/python -u scratch_qual200.py; echo "DONE qual"
