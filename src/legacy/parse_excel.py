"""
Parse 'Gross Description' trong RCC xlsx -> kích thước KHỐI U, rồi join vào
metadata.csv (theo bệnh nhân).

Quan trọng:
  - Số cm lấy ra là kích thước VÙNG TỔN THƯƠNG (mass/tumor), KHÔNG phải lát cắt.
  - Dùng để sanity-check DIỆN TÍCH MASK KHỐI U (có sau), không so với slice_area_cm2.
  - Kích u đo bằng caliper trên mặt cắt -> chỉ là tín hiệu QC, đúng khi ảnh đúng mặt cắt.

Khoá join: chuẩn hoá bỏ ký tự không phải chữ-số, viết hoa
  Excel 'SS2363742'  -> 'SS2363742'
  file  'SS23-63742' -> 'SS2363742'

Ví dụ:
  python parse_excel.py --xlsx "data/20241212/RCC 20241212.xlsx" \
      --metadata processed/metadata.csv --out processed
"""
import argparse
import csv
import os
import re

import openpyxl

# từ khoá chỉ khối u (không lấy 'kidney', 'nephrectomy' = kích thận/bệnh phẩm)
TUMOR_KW = re.compile(r"\b(mass|tumou?r|lesion|nodule)\b", re.I)
# D x D [x D] cm  (bắt 2 hoặc 3 chiều), tránh khoảng cách rìa (số đơn lẻ)
DIM = re.compile(
    r"(\d+(?:\.\d+)?)\s*[x×X]\s*(\d+(?:\.\d+)?)"
    r"(?:\s*[x×X]\s*(\d+(?:\.\d+)?))?\s*cm",
    re.I,
)
WINDOW = 90  # số ký tự sau từ khoá để tìm kích thước


def canon(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(key)).upper()


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("_x000D_", " ")).strip()


def parse_mass(gross: str):
    """
    Trả về dict: dims(list, giảm dần), area_cm2(rect 2 chiều lớn nhất),
    n_candidates, flag, snippet.
    """
    g = clean(gross)
    cands = []  # (dims_sorted_desc, snippet)
    for m in TUMOR_KW.finditer(g):
        seg = g[m.start(): m.end() + WINDOW]
        dm = DIM.search(seg)
        if dm:
            dims = [float(x) for x in dm.groups() if x is not None]
            dims.sort(reverse=True)
            cands.append((dims, g[m.start(): m.start() + 60]))
    if not cands:
        # fallback: từ khoá có nhưng không có kích thước trong cửa sổ
        flag = "no_dim" if TUMOR_KW.search(g) else "no_keyword"
        return {"dims": [], "area_cm2": None, "n": 0, "flag": flag, "snippet": g[:60]}

    # chọn khối u LỚN NHẤT (tích 2 chiều lớn nhất) làm u chính
    def area2(d):
        return d[0] * (d[1] if len(d) > 1 else d[0])
    cands.sort(key=lambda c: area2(c[0]), reverse=True)
    dims, snip = cands[0]
    flag = "ok" if len(cands) == 1 else "multi_mass"
    long_, short_ = dims[0], (dims[1] if len(dims) > 1 else dims[0])
    return {
        "dims": dims, "area_cm2": round(long_ * short_, 2),
        "n": len(cands), "flag": flag, "snippet": snip,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="data/20241212/RCC 20241212.xlsx")
    ap.add_argument("--metadata", default="processed/metadata.csv")
    ap.add_argument("--out", default="processed")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    wb = openpyxl.load_workbook(args.xlsx, read_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(min_row=2, values_only=True))

    by_key = {}        # canon -> parsed record
    parsed_rows = []   # để xuất excel_parsed.csv
    dup = 0
    for r in rows:
        path_no = r[0]
        if path_no is None:
            continue
        key = canon(path_no)
        info = parse_mass(r[3] or "")
        rec = {
            "path_no": str(path_no).strip(), "canon": key,
            "diagnosis": clean(r[2])[:80],
            "mass_long_cm": info["dims"][0] if info["dims"] else "",
            "mass_short_cm": (info["dims"][1] if len(info["dims"]) > 1 else
                              (info["dims"][0] if info["dims"] else "")),
            "mass_dims": "x".join(str(d) for d in info["dims"]),
            "mass_area_cm2": info["area_cm2"] if info["area_cm2"] else "",
            "n_mass": info["n"], "parse_flag": info["flag"],
            "mass_snippet": info["snippet"],
        }
        if key in by_key:
            dup += 1
        by_key[key] = rec
        parsed_rows.append(rec)

    # thống kê chất lượng parse
    from collections import Counter
    flags = Counter(r["parse_flag"] for r in parsed_rows)
    print(f"Excel: {len(parsed_rows)} dòng | khoá trùng: {dup}")
    print("Parse flags:", dict(flags))
    ok = flags.get("ok", 0) + flags.get("multi_mass", 0)
    print(f"Lấy được kích u: {ok}/{len(parsed_rows)} "
          f"({100*ok/max(1,len(parsed_rows)):.0f}%)")

    # xuất bản parse để duyệt tay
    pe_path = os.path.join(args.out, "excel_parsed.csv")
    with open(pe_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(parsed_rows[0].keys()))
        w.writeheader(); w.writerows(parsed_rows)
    print(f"-> {pe_path}")

    # join vào metadata.csv nếu có
    if os.path.isfile(args.metadata):
        with open(args.metadata) as f:
            meta = list(csv.DictReader(f))
        add_cols = ["mass_long_cm", "mass_short_cm", "mass_dims",
                    "mass_area_cm2", "n_mass", "parse_flag"]
        matched = 0
        for row in meta:
            key = canon(row.get("patient_id", ""))
            rec = by_key.get(key)
            if rec:
                matched += 1
                for c in add_cols:
                    row[c] = rec[c]
            else:
                for c in add_cols:
                    row[c] = ""
                row["parse_flag"] = "no_excel_match"
        out_path = os.path.join(args.out, "metadata_enriched.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
            w.writeheader(); w.writerows(meta)
        print(f"Join: {matched}/{len(meta)} ảnh khớp Excel -> {out_path}")
    else:
        print(f"[bỏ qua join] không thấy {args.metadata} (chạy preprocess.py trước)")


if __name__ == "__main__":
    main()
