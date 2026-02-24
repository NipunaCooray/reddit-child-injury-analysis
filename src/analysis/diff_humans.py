#!/usr/bin/env python3
"""
Diff two human-coded CSVs and list disagreements, with summaries and confusion pairs.
Context columns are written UNTRIMMED (full title/selftext/etc).

HOW TO RUN (examples):

# Minimal (default fields; title+permalink as context)
python src/analysis/diff_humans.py \
  --coder-a data/interim/gold/sample_annotator_NC.csv \
  --coder-b data/interim/gold/sample_annotator_SP.csv

# Specify output directory and include selftext in the context
python src/analysis/diff_humans.py \
  --coder-a data/interim/gold/sample_annotator_NC.csv \
  --coder-b data/interim/gold/sample_annotator_SP.csv \
  --out-dir reports/diffs \
  --context title,permalink,selftext

# Compare only selected fields and ignore blank labels
python src/analysis/diff_humans.py \
  --coder-a data/interim/gold/sample_annotator_A.csv \
  --coder-b data/interim/gold/sample_annotator_B.csv \
  --fields is_injury_event,age_group \
  --ignore-blanks

Outputs (to --out-dir, default: reports/diffs):
- diff_summary.csv              (per-field counts and % disagreement)
- diff_details.csv              (row-wise disagreements with full context columns)
- confusions_<field>.csv        (per-field A→B pair frequencies)
- orphans_A.csv / orphans_B.csv (IDs present only in one file), if any
"""

import csv
import argparse
from pathlib import Path
from collections import Counter, defaultdict

# ---------- Default target fields (same family as kappa_humans) ----------
FIELDS_DEFAULT = [
    "is_injury_event",
    "mechanism_of_injury",
    "nature_of_injury",
    "body_region",
    "er_or_hospital_mentioned",
    "age_group",
]
BOOL_FIELDS = {"is_injury_event", "er_or_hospital_mentioned"}

# ---------- Normalization (mirror kappa_humans behavior) ----------
def _norm_bool(x: str) -> str:
    s = (x or "").strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return "true"
    if s in {"false", "0", "no", "n"}:
        return "false"
    return ""  # treat unknown/blank as missing for blanks-only logic

def _norm_cat(x: str) -> str:
    return (x or "").strip().lower()

# ---------- IO ----------
def load_by_post_id(path: Path, fields: list[str], context_cols: list[str]) -> dict[str, dict]:
    """Return dict[post_id] -> { field: normalized_value, <context cols>: original (UNTRIMMED) }"""
    data: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as fp:
        r = csv.DictReader(fp)
        for row in r:
            pid = row.get("post_id") or row.get("id")
            if not pid:
                continue
            rec = {}
            # normalized fields
            for f in fields:
                val = row.get(f, "")
                if f in BOOL_FIELDS:
                    rec[f] = _norm_bool(val)
                else:
                    rec[f] = _norm_cat(val)
            # raw context (full text, no trimming)
            for c in context_cols:
                rec[c] = row.get(c, "")
            data[pid] = rec
    return data

# ---------- Main diff logic ----------
def diff_coders(A: dict, B: dict, fields: list[str], ignore_blanks: bool, context_cols: list[str]):
    common = sorted(set(A.keys()) & set(B.keys()))
    only_A = sorted(set(A.keys()) - set(B.keys()))
    only_B = sorted(set(B.keys()) - set(A.keys()))

    # per-field confusion counts: (a_val, b_val) -> count
    confusions: dict[str, Counter] = {f: Counter() for f in fields}
    # per-field totals and disagreements
    compared = defaultdict(int)
    disagreed = defaultdict(int)

    # detailed rows
    details_rows = []

    for pid in common:
        recA, recB = A[pid], B[pid]
        for f in fields:
            va, vb = recA.get(f, ""), recB.get(f, "")
            if ignore_blanks and (va == "" or vb == ""):
                continue

            compared[f] += 1
            if va != vb:
                disagreed[f] += 1
                row = {
                    "post_id": pid,
                    "field": f,
                    "coder_a": va,
                    "coder_b": vb,
                }
                # include full context columns (untrimmed). Prefer A's value; fall back to B's if empty.
                for c in context_cols:
                    row[c] = recA.get(c, "") if recA.get(c, "") != "" else recB.get(c, "")
                details_rows.append(row)

            confusions[f][(va, vb)] += 1

    return {
        "common": common,
        "only_A": only_A,
        "only_B": only_B,
        "details": details_rows,
        "compared": dict(compared),
        "disagreed": dict(disagreed),
        "confusions": confusions,
        "context_cols": context_cols,
    }

# ---------- Writing helpers ----------
def write_summary(out_dir: Path, compared: dict, disagreed: dict):
    out = out_dir / "diff_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["field", "n_compared", "n_disagreed", "pct_disagree"])
        w.writeheader()
        for f in sorted(compared.keys()):
            n = compared[f]
            d = disagreed.get(f, 0)
            pct = (d / n) * 100 if n else 0.0
            w.writerow({"field": f, "n_compared": n, "n_disagreed": d, "pct_disagree": f"{pct:.2f}"})
    return out

def write_details(out_dir: Path, details_rows: list[dict], context_cols: list[str]):
    out = out_dir / "diff_details.csv"
    fields = ["post_id", "field", "coder_a", "coder_b"] + context_cols
    with open(out, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(details_rows)
    return out

def write_confusions(out_dir: Path, confusions: dict[str, Counter]):
    paths = {}
    for f, counter in confusions.items():
        out = out_dir / f"confusions_{f}.csv"
        with open(out, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=["coder_a_value", "coder_b_value", "count"])
            w.writeheader()
            for (va, vb), cnt in counter.most_common():
                w.writerow({"coder_a_value": va, "coder_b_value": vb, "count": cnt})
        paths[f] = out
    return paths

def write_orphans(out_dir: Path, only_A: list[str], only_B: list[str]):
    paths = {}
    if only_A:
        p = out_dir / "orphans_A.csv"
        with open(p, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=["post_id"])
            w.writeheader()
            for pid in only_A:
                w.writerow({"post_id": pid})
        paths["A"] = p
    if only_B:
        p = out_dir / "orphans_B.csv"
        with open(p, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=["post_id"])
            w.writeheader()
            for pid in only_B:
                w.writerow({"post_id": pid})
        paths["B"] = p
    return paths

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Diff two human-coded CSVs and list disagreements (no trimming).")
    ap.add_argument("--coder-a", required=True, help="CSV from annotator A")
    ap.add_argument("--coder-b", required=True, help="CSV from annotator B")
    ap.add_argument("--out-dir", default="reports/diffs", help="Directory to write diff outputs")
    ap.add_argument("--fields", default=",".join(FIELDS_DEFAULT),
                    help="Comma-separated list of fields to compare")
    ap.add_argument("--context", default="title,permalink",
                    help="Comma-separated context columns to include in details (e.g., title,selftext,permalink)")
    ap.add_argument("--ignore-blanks", action="store_true",
                    help="Skip comparisons where either coder left the field blank")
    return ap.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    context_cols = [c.strip() for c in args.context.split(",") if c.strip()]

    A = load_by_post_id(Path(args.coder_a), fields, context_cols)
    B = load_by_post_id(Path(args.coder_b), fields, context_cols)

    res = diff_coders(A, B, fields, args.ignore_blanks, context_cols)

    out_summary = write_summary(out_dir, res["compared"], res["disagreed"])
    out_details = write_details(out_dir, res["details"], context_cols)
    out_conf = write_confusions(out_dir, res["confusions"])
    out_orphans = write_orphans(out_dir, res["only_A"], res["only_B"])

    print(f"Compared posts (overlap): {len(res['common'])}")
    print(f"Only in A: {len(res['only_A'])} | Only in B: {len(res['only_B'])}")
    print("\nPer-field disagreement:")
    for f in fields:
        n = res["compared"].get(f, 0)
        d = res["disagreed"].get(f, 0)
        pct = (d / n) * 100 if n else 0.0
        print(f"  {f:>24}  n={n:<4}  disagreed={d:<4}  {pct:5.1f}%")

    print(f"\nWrote: {out_summary}")
    print(f"Wrote: {out_details}")
    for f, p in out_conf.items():
        print(f"Wrote: {p}")
    for k, p in out_orphans.items():
        print(f"Wrote: {p}  (orphans_{k})")

if __name__ == "__main__":
    main()
