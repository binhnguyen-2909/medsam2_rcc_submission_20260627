"""
Sinh THỨ TỰ GÁN ƯU TIÊN cho app QUEUE (active-learning bằng diversity sampling):
- chỉ ảnh MẶT-CẮT chưa gán (processed/cut_surface_filter.csv, trừ done/skipped)
- LOẠI ảnh thuộc bệnh nhân trong test đóng băng (gán sẽ phí vì không vào train được)
- round-robin theo BỆNH NHÂN (mỗi vòng lấy 1 ảnh/bệnh nhân) -> phủ nhiều bệnh
  nhân trước = đa dạng nhất; trong 1 bệnh nhân ưu tiên cut_score cao.

-> labels/queue_order.json (list stem). app QUEUE đọc file này nếu có.
  python make_queue_order.py
"""
import csv
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(ROOT, "labels")


def patient_of(s):
    return s.split("^")[0]


def hkey(p):
    return hashlib.md5(p.encode()).hexdigest()


def load_json(name, default):
    p = os.path.join(LABELS, name)
    return json.load(open(p)) if os.path.isfile(p) else default


def main():
    done = set(load_json("done.json", []))
    skip = set(load_json("skipped.json", []))
    test = load_json("test_frozen.json", {"test_patients": []})
    test_pats = set(test.get("test_patients", []))

    cut = {}
    fp = os.path.join(ROOT, "processed/cut_surface_filter.csv")
    for r in csv.DictReader(open(fp)):
        if r["is_cut_surface"] != "1":
            continue
        s = r["stem"]
        if s in done or s in skip or patient_of(s) in test_pats:
            continue
        cut[s] = float(r["cut_surface_score"] or 0)

    by_pat = {}
    for s, sc in cut.items():
        by_pat.setdefault(patient_of(s), []).append((sc, s))
    for p in by_pat:
        by_pat[p].sort(reverse=True)              # cut_score cao trước

    pats = sorted(by_pat, key=hkey)               # thứ tự bệnh nhân ổn định
    order = []
    i = 0
    while any(by_pat.values()):
        p = pats[i % len(pats)]
        if by_pat[p]:
            order.append(by_pat[p].pop(0)[1])
        i += 1
        if i > len(cut) * 3:                      # chốt an toàn
            break

    json.dump(order, open(os.path.join(LABELS, "queue_order.json"), "w"))
    print(f"Ưu tiên {len(order)} ảnh mặt-cắt chưa gán "
          f"({len(by_pat)} bệnh nhân, đã loại {len(test_pats)} bệnh nhân test).")
    print("-> labels/queue_order.json (app QUEUE sẽ dùng thứ tự này)")


if __name__ == "__main__":
    main()
