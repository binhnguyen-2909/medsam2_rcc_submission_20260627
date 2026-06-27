"""
Gộp kết quả 1 vòng (results/finetune_last.json) vào đường cong human-in-the-loop
và vẽ lại: Dice (zero-shot & fine-tuned) theo SỐ ẢNH TRAIN.

Cho biết: thêm nhãn có giúp không, và ở mốc N nào fine-tune mới VƯỢT zero-shot.
  python update_curve.py
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
CURVE = os.path.join(ROOT, "results/loop_curve.csv")
LAST = os.path.join(ROOT, "results/finetune_last.json")


def main():
    r = json.load(open(LAST))
    rows = []
    if os.path.isfile(CURVE):
        rows = [x for x in csv.DictReader(open(CURVE))
                if int(x["n_train"]) != r["n_train"]]   # ghi đè nếu trùng N
    rows.append({"n_train": r["n_train"], "n_test": r["n_test"],
                 "zero_shot_dice": r["zero_shot_dice"], "best_ft_dice": r["best_ft_dice"]})
    rows.sort(key=lambda x: int(x["n_train"]))
    with open(CURVE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_train", "n_test", "zero_shot_dice", "best_ft_dice"])
        w.writeheader(); w.writerows(rows)

    N = [int(x["n_train"]) for x in rows]
    zs = [float(x["zero_shot_dice"]) for x in rows]
    ft = [float(x["best_ft_dice"]) for x in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(N, zs, "o--", label="zero-shot (không train)", color="gray")
    plt.plot(N, ft, "o-", label="fine-tuned (best)", color="C0")
    for x, y in zip(N, ft):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=8)
    plt.xlabel("Số ảnh train (nhãn người)")
    plt.ylabel("Test Dice (12 ảnh test cố định)")
    plt.title("Human-in-the-loop: Dice theo số nhãn")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "results/loop_curve.png"), dpi=110)

    print(f"Đường cong ({len(rows)} mốc):")
    for x in rows:
        flag = " <- FT thắng" if float(x["best_ft_dice"]) > float(x["zero_shot_dice"]) else ""
        print(f"  N_train={x['n_train']:>4}  zero-shot={x['zero_shot_dice']}  "
              f"fine-tuned={x['best_ft_dice']}{flag}")
    print("-> results/loop_curve.csv , results/loop_curve.png")


if __name__ == "__main__":
    main()
