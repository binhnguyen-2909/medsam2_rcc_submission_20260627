"""
Chia tập nhãn (55 ảnh đã accept) thành TRAIN / VAL theo BỆNH NHÂN.
KHÔNG để 1 bệnh nhân nằm cả 2 phía (tránh rò rỉ: 2 ảnh/người rất giống nhau).

- Nguồn: labels/done.json (ảnh đã accept) ∩ có file mask labels/masks/<stem>.png
- Deterministic: thứ tự bệnh nhân theo md5(patient_id) -> tái lập y hệt mọi lần chạy,
  KHÔNG dùng random (seed nào cũng vậy).
- Tỉ lệ val ~ VAL_FRAC theo SỐ ẢNH (gộp nguyên cụm ảnh của từng bệnh nhân).

Xuất:
  labels/split.json   {train:[stems], val:[stems], train_patients, val_patients}
  labels/split.csv    stem,patient_id,split
  In tóm tắt + kiểm tra không trùng bệnh nhân.

Chạy: python split_dataset.py [--val_frac 0.2]
"""
import argparse
import csv
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(ROOT, "labels")


def load_done():
    p = os.path.join(LABELS, "done.json")
    return set(json.load(open(p))) if os.path.isfile(p) else set()


def patient_of(stem):
    return stem.split("^")[0]


def has_mask(stem):
    return os.path.isfile(os.path.join(LABELS, "masks", stem + ".png"))


def hkey(pid):
    return hashlib.md5(pid.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_frac", type=float, default=0.2)
    args = ap.parse_args()

    done = load_done()
    stems = sorted(s for s in done if has_mask(s))
    missing = sorted(s for s in done if not has_mask(s))
    if missing:
        print(f"[warn] {len(missing)} ảnh done nhưng thiếu mask -> bỏ:", missing[:3], "...")

    # gom ảnh theo bệnh nhân
    by_pat = {}
    for s in stems:
        by_pat.setdefault(patient_of(s), []).append(s)

    # --- TEST ĐÓNG BĂNG (human-in-the-loop): test cố định mọi vòng ---
    frozen_p = os.path.join(LABELS, "test_frozen.json")
    if os.path.isfile(frozen_p):
        fr = json.load(open(frozen_p))
        val = sorted(s for s in fr["test"] if has_mask(s))
        val_pats = sorted(set(map(patient_of, val)))
        # train = MỌI ảnh đã gán còn lại, LOẠI bệnh nhân thuộc test (chống rò rỉ)
        test_pat_set = set(val_pats)
        train = sorted(s for s in stems
                       if patient_of(s) not in test_pat_set and s not in set(val))
        train_pats = sorted(set(map(patient_of, train)))
        print(f"[test ĐÓNG BĂNG] {len(val)} ảnh test cố định; train nhận nhãn mới.")
    else:
        pats = sorted(by_pat, key=hkey)      # thứ tự ổn định, độc lập tên
        total = len(stems)
        target_val = round(args.val_frac * total)
        val_pats, val_n = [], 0
        for p in pats:
            if val_n < target_val:
                val_pats.append(p)
                val_n += len(by_pat[p])
        train_pats = [p for p in pats if p not in set(val_pats)]
        train = sorted(s for p in train_pats for s in by_pat[p])
        val = sorted(s for p in val_pats for s in by_pat[p])

    # kiểm tra rò rỉ
    overlap = set(map(patient_of, train)) & set(map(patient_of, val))
    assert not overlap, f"RÒ RỈ bệnh nhân giữa train/val: {overlap}"

    total = len(train) + len(val)
    n_pat = len(set(map(patient_of, train + val)))
    out = {
        "val_frac_target": args.val_frac,
        "n_total": total, "n_train": len(train), "n_val": len(val),
        "n_patients": n_pat,
        "train_patients": sorted(train_pats), "val_patients": sorted(val_pats),
        "train": train, "val": val,
    }
    json.dump(out, open(os.path.join(LABELS, "split.json"), "w"), indent=1)

    with open(os.path.join(LABELS, "split.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "patient_id", "split"])
        for s in train:
            w.writerow([s, patient_of(s), "train"])
        for s in val:
            w.writerow([s, patient_of(s), "val"])

    print(f"Tổng dùng: {total} ảnh / {n_pat} bệnh nhân")
    print(f"TRAIN: {len(train)} ảnh / {len(train_pats)} bệnh nhân")
    print(f"TEST : {len(val)} ảnh / {len(val_pats)} bệnh nhân")
    print("Không rò rỉ bệnh nhân ✔")
    print("-> labels/split.json , labels/split.csv")


if __name__ == "__main__":
    main()
