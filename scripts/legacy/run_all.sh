#!/usr/bin/env bash
# =====================================================================
# Tự động chạy nốt phần CHUẨN BỊ DỮ LIỆU trên server, CHỊU ĐƯỢC tắt máy/
# đăng xuất (chạy bằng nohup/setsid -> không phụ thuộc phiên SSH).
#
#   Chuỗi việc (đều CPU, hoàn tất trọn vẹn, KHÔNG cần người):
#     0. Chờ preprocess.py đang chạy xong (metadata.csv đủ 1392 ảnh)
#     1. split_dataset.py        -> chia train/val theo bệnh nhân
#     2. filter_cut_surface.py   -> lọc ảnh mặt-cắt vs mặt-ngoài
#     3. area_sanity.py          -> sanity-check diện tích u (cm²)
#     4. Dựng CHECKPOINT.md      -> gộp kết quả + 2 việc cần người
#
#   KHÔNG đưa vào: zero-shot đề xuất mask (box->mask cần box do người vẽ),
#   train/fine-tune (chờ quyết định ở checkpoint).
#
# CÁCH CHẠY (sống sót tắt máy):
#     cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
#     setsid nohup bash run_all.sh > /dev/null 2>&1 &
#   Theo dõi:  tail -f run_all.log   |   xem trạng thái: cat run_all.status
# =====================================================================
set -u
cd "/home/hvusynh2/nguyenduong/MedSAM2 a/MedSAM2"
PY=/home/hvusynh2/conda_envs/medsam2_anno/bin/python
LOG=run_all.log
ST=run_all.status

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
fail() { echo "ERROR: $*" | tee -a "$LOG"; echo "ERROR" > "$ST"; exit 1; }

: > "$LOG"
echo "RUNNING" > "$ST"
say "=== run_all bắt đầu (pid $$) ==="

# 0. chờ preprocess xong: metadata.csv phải có >=1392 dòng dữ liệu ----------
say "[0/4] Chờ preprocess (cần 1392 ảnh trong processed/metadata.csv)..."
for i in $(seq 1 720); do            # tối đa ~2h (720 x 10s)
  n=$(( $(wc -l < processed/metadata.csv 2>/dev/null || echo 1) - 1 ))
  if [ "$n" -ge 1392 ]; then say "    preprocess XONG ($n ảnh)."; break; fi
  if ! pgrep -f "[p]reprocess.py" >/dev/null && [ "$n" -lt 1392 ]; then
    # tiến trình chết mà chưa đủ -> chờ thêm 1 nhịp rồi kiểm tra lại file
    sleep 5
    n=$(( $(wc -l < processed/metadata.csv 2>/dev/null || echo 1) - 1 ))
    [ "$n" -ge 1392 ] && { say "    preprocess XONG ($n ảnh)."; break; }
    fail "preprocess không còn chạy nhưng metadata mới có $n/1392 ảnh."
  fi
  sleep 10
done

# 1. split train/val ------------------------------------------------------
say "[1/4] split_dataset.py"
$PY split_dataset.py 2>&1 | tee out_split.txt | sed 's/^/    /' | tee -a "$LOG" \
  || fail "split_dataset.py lỗi"

# 2. lọc mặt-cắt ----------------------------------------------------------
say "[2/4] filter_cut_surface.py"
$PY filter_cut_surface.py 2>&1 | tee out_filter.txt | sed 's/^/    /' | tee -a "$LOG" \
  || fail "filter_cut_surface.py lỗi"

# 3. sanity cm² -----------------------------------------------------------
say "[3/4] area_sanity.py"
$PY area_sanity.py 2>&1 | tee out_area.txt | sed 's/^/    /' | tee -a "$LOG" \
  || fail "area_sanity.py lỗi"

# 4. dựng CHECKPOINT.md ---------------------------------------------------
say "[4/4] Dựng CHECKPOINT.md"
{
  echo "# CHECKPOINT — chuẩn bị dữ liệu (tự sinh bởi run_all.sh)"
  echo
  echo "_Sinh lúc: $(date '+%F %T')_"
  echo
  echo "## 1) Split train/val (theo bệnh nhân, 0 rò rỉ)"
  echo '```'; cat out_split.txt; echo '```'
  echo "Chi tiết: \`labels/split.json\`, \`labels/split.csv\`"
  echo
  echo "## 2) Lọc ảnh mặt-cắt vs mặt-ngoài (toàn 1392)"
  echo '```'; cat out_filter.txt; echo '```'
  echo "Chi tiết: \`processed/cut_surface_filter.csv\` — QC ngưỡng: \`results/cut_filter_montage.jpg\`"
  echo
  echo "## 3) Sanity-check diện tích khối u (cm², chuẩn hoá px/cm)"
  echo '```'; cat out_area.txt; echo '```'
  echo
  echo "## 4) Hai việc BẢN CHẤT cần con người (chưa làm)"
  echo "- **(a) Xác nhận thước:** liếc vài ảnh trong \`processed/qc/\` xem 1 vạch thước = 1cm không."
  echo "  Nếu không phải 1cm, chạy lại preprocess với \`--tick_cm <đúng>\` thì px/cm & cm² mới chuẩn."
  echo "- **(b) Quyết hướng model:** zero-shot SAM2.1 đã đủ, hay fine-tune trên 55 nhãn,"
  echo "  hay gán thêm nhãn trước (box->mask cần người vẽ box)."
  echo
  echo "## Đầu ra đã tạo"
  echo "- \`labels/split.json\`, \`labels/split.csv\`"
  echo "- \`processed/metadata.csv\` (px/cm, ruler_conf, cut_surface_score x1392)"
  echo "- \`processed/cut_surface_filter.csv\`"
  echo "- \`results/cut_filter_montage.jpg\`"
} > CHECKPOINT.md

say "=== HOÀN TẤT. Xem CHECKPOINT.md ==="
echo "DONE" > "$ST"
