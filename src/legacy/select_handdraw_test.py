"""
Chọn ~50 ảnh cho TEST SET VẼ TAY ĐỘC LẬP (phá vòng GT-do-SAM).
Pool = test12 ∪ eval200 (đều held-out khỏi detector train). Bắt buộc gồm cả 12 frozen.
Phân tầng theo số u (đếm connected-component MASK HIỆN CÓ — chỉ để chọn ảnh, mask vẽ tay vẫn độc lập)
để đủ cả 1u và >1u. Deterministic theo md5(stem). Xuất labels_handdraw/select.json.
"""
import os, sys, json, hashlib, numpy as np, cv2
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT)
N_TARGET = 50
MIN_FRAC = 0.002  # bỏ mảnh nhiễu < 0.2% diện tích ảnh khi đếm u

def ncomp(stem):
    p = f"labels/masks/{stem}.png"
    if not os.path.isfile(p): return 1
    m = cv2.imread(p, 0) > 127
    if m.sum() == 0: return 1
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    thr = MIN_FRAC * m.size
    return max(1, int((stats[1:, cv2.CC_STAT_AREA] >= thr).sum()))

def h(s): return hashlib.md5(s.encode()).hexdigest()

sp = json.load(open("labels/split_detector.json"))
test12, eval200 = sp["test12"], sp["eval200"]
pool = list(dict.fromkeys(test12 + eval200))  # giữ thứ tự, loại trùng
no = {s: ncomp(s) for s in pool}

forced = list(test12)                              # luôn gồm 12 frozen
rest = [s for s in pool if s not in set(forced)]
single = sorted([s for s in rest if no[s] <= 1], key=h)
multi  = sorted([s for s in rest if no[s] >  1], key=h)

# 12 frozen + bù tới 50, ưu tiên cân bằng: nhắm ~ một nửa >1u trong số bù
n_need = N_TARGET - len(forced)
n_multi = min(len(multi), n_need // 2 + n_need % 2)   # nhỉnh phía multi (hiếm hơn, quan trọng)
n_single = n_need - n_multi
n_single = min(n_single, len(single))
n_multi = min(n_need - n_single, len(multi))
pick = forced + single[:n_single] + multi[:n_multi]
pick = list(dict.fromkeys(pick))

sel = sorted(pick, key=h)
cnt1 = sum(1 for s in sel if no[s] <= 1); cntm = len(sel) - cnt1
os.makedirs("labels_handdraw", exist_ok=True)
json.dump({"stems": sel, "n_objects_hint": {s: no[s] for s in sel},
           "note": "n_objects_hint = #component MASK SAM cũ, CHỈ để chọn ảnh; mask vẽ tay độc lập"},
          open("labels_handdraw/select.json", "w"), indent=1, ensure_ascii=False)
print(f"Đã chọn {len(sel)} ảnh: {cnt1} đơn-u + {cntm} đa-u (gồm đủ 12 frozen: "
      f"{all(s in sel for s in test12)})")
print(f"Pool: test12={len(test12)} eval200={len(eval200)} | multi khả dụng={len(multi)} single={len(single)}")
print("-> labels_handdraw/select.json")
