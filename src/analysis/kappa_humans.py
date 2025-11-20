#!/usr/bin/env python3
"""
Cohen's kappa between two human coder CSVs (your revised schema).

Expected columns in each CSV (aligned by post_id):
post_id, title, selftext, permalink, created_utc, subreddit,
is_injury_event, mechanism_of_injury, nature_of_injury, body_region,
er_or_hospital_mentioned, age_group, rationale_short, coder_id
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
    ap = argparse.ArgumentParser(description="Compute Cohen's kappa between two human coders.")
    ap.add_argument("--coder-a", required=True, help="CSV from annotator A")
    ap.add_argument("--coder-b", required=True, help="CSV from annotator B")
    args = ap.parse_args()

    A = load_annotations(Path(args.coder_a))
    B = load_annotations(Path(args.coder_b))

    common = sorted(set(A.keys()) & set(B.keys()))
    if not common:
        raise SystemExit("No overlapping post_id between the two CSVs.")

    print(f"Comparing {len(common)} posts\n")
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
            print(f"{field:>22}:  kappa=nan  agreement=nan  n=0 (no overlapping non-empty labels)")
            continue

        kappa = cohen_kappa_score(y_true, y_pred)
        agreement = sum(1 for i in range(n) if y_true[i] == y_pred[i]) / n
        print(f"{field:>22}:  kappa={kappa:.3f}  agreement={agreement:.3f}  n={n}")

if __name__ == "__main__":
    main()
