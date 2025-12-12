#!/usr/bin/env python3
"""
Cohen's kappa between two human coder CSVs (revised schema).

Expected columns in each CSV (aligned by post_id):
post_id, title, selftext, permalink, created_utc, subreddit,
is_injury_event, mechanism_of_injury, nature_of_injury, body_region,
er_or_hospital_mentioned, age_group, rationale_short, coder_id

Usage:
  python src/analysis/kappa_humans.py \
    --coder-a data/interim/gold/sample_annotator_NC.csv \
    --coder-b data/interim/gold/sample_annotator_SP.csv \
    --out reports/metrics/kappa_summary.csv
"""

import csv
import argparse
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

# Fields to evaluate (exclude rationale_short, coder_id)
FIELDS = [
    "is_injury_event",
    "mechanism_of_injury",
    "nature_of_injury",
    "body_region",
    "er_or_hospital_mentioned",
    "age_group",
]

BOOL_FIELDS = {"is_injury_event", "er_or_hospital_mentioned"}

def _norm_bool(x: str) -> str:
    s = (x or "").strip().lower()
    if s in {"true","1","yes","y"}: return "true"
    if s in {"false","0","no","n"}: return "false"
    return ""  # treat unknown/blank as missing

def _norm_cat(x: str) -> str:
    return (x or "").strip().lower()

def load_annotations(path: Path) -> dict:
    data = {}
    with open(path, "r", encoding="utf-8") as fp:
        r = csv.DictReader(fp)
        for row in r:
            pid = row.get("post_id") or row.get("id")
            if not pid:
                continue
            rec = {}
            for f in FIELDS:
                val = row.get(f, "")
                if f in BOOL_FIELDS:
                    rec[f] = _norm_bool(val)
                else:
                    rec[f] = _norm_cat(val)
            data[pid] = rec
    return data

def main():
    ap = argparse.ArgumentParser(description="Compute Cohen's kappa between two human coder CSVs.")
    ap.add_argument("--coder-a", required=True, help="CSV from annotator A")
    ap.add_argument("--coder-b", required=True, help="CSV from annotator B")
    ap.add_argument("--out", default="reports/metrics/kappa_summary.csv",
                    help="Output CSV path for summary metrics")
    args = ap.parse_args()

    A = load_annotations(Path(args.coder_a))
    B = load_annotations(Path(args.coder_b))

    common = sorted(set(A.keys()) & set(B.keys()))
    if not common:
        raise SystemExit("No overlapping post_id between the two CSVs.")

    print(f"Comparing {len(common)} posts\n")

    rows_out = []
    for field in FIELDS:
        y_true, y_pred = [], []
        for pid in common:
            va, vb = A[pid].get(field, ""), B[pid].get(field, "")
            # skip if either coder left blank/unknown for this field
            if va == "" or vb == "":
                continue
            y_true.append(va)
            y_pred.append(vb)

        n = len(y_true)
        if n == 0:
            kappa = float("nan")
            agreement = float("nan")
            print(f"{field:>22}:  kappa=nan  agreement=nan  n=0 (no overlapping non-empty labels)")
        else:
            kappa = cohen_kappa_score(y_true, y_pred)
            agreement = sum(1 for i in range(n) if y_true[i] == y_pred[i]) / n
            print(f"{field:>22}:  kappa={kappa:.3f}  agreement={agreement:.3f}  n={n}")

        rows_out.append({"field": field, "kappa": kappa, "agreement": agreement, "n": n})

    # Write summary CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["field","kappa","agreement","n"])
        w.writeheader()
        for r in rows_out:
            # format numbers for easier spreadsheet use
            row = dict(r)
            if isinstance(row["kappa"], float):
                row["kappa"] = f"{row['kappa']:.6f}" if row["n"] else ""
            if isinstance(row["agreement"], float):
                row["agreement"] = f"{row['agreement']:.6f}" if row["n"] else ""
            w.writerow(row)

    print(f"\nSaved summary -> {out_path}")

if __name__ == "__main__":
    main()
