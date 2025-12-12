#!/usr/bin/env python3
"""
Evaluate model vs. adjudicated gold using scikit-learn:
- accuracy, macro precision/recall/F1, Cohen's kappa
- per-field confusion matrices (CSV)
- summary CSV

Gold CSV schema:
  post_id,title,selftext,permalink,created_utc,subreddit,
  is_injury_event,mechanism_of_injury,nature_of_injury,body_region,
  er_or_hospital_mentioned,age_group,rationale_short,coder_id

Model CSV schema (from classifier):
  id,subreddit,created_utc,permalink,title,
  is_injury_event,mechanism_of_injury,nature_of_injury,body_region,
  er_or_hospital_mentioned,age_group,rationale_short

Usage:
  python src/analysis/evaluate_labels.py \
    --gold data/interim/gold/sample_adjudicated.csv \
    --model data/interim/reddit_child_injury_YYYYMMDD_labels.csv \
    --outdir reports/metrics
"""

import csv
from pathlib import Path
import argparse

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

EVAL_FIELDS = [
    "is_injury_event",
    "mechanism_of_injury",
    "nature_of_injury",
    "body_region",
    "er_or_hospital_mentioned",
    "age_group",
]

BOOL_FIELDS = {"is_injury_event", "er_or_hospital_mentioned"}

def norm_bool(x: str) -> str:
    s = (x or "").strip().lower()
    if s in {"true","1","yes","y"}: return "true"
    if s in {"false","0","no","n"}: return "false"
    return ""  # treat missing/unknown as empty

def norm_cat(x: str) -> str:
    return (x or "").strip().lower()

def load_gold(path: Path) -> dict:
    gold = {}
    with open(path, "r", encoding="utf-8") as fp:
        r = csv.DictReader(fp)
        for row in r:
            pid = row.get("post_id")
            if not pid:
                continue
            rec = {}
            for f in EVAL_FIELDS:
                v = row.get(f, "")
                rec[f] = norm_bool(v) if f in BOOL_FIELDS else norm_cat(v)
            gold[pid] = rec
    return gold

def load_model(path: Path) -> dict:
    model = {}
    with open(path, "r", encoding="utf-8") as fp:
        r = csv.DictReader(fp)
        model_cols = set(r.fieldnames or [])
        for row in r:
            pid = row.get("post_id") or row.get("id")
            if not pid:
                continue
            rec = {}
            for f in EVAL_FIELDS:
                if f not in model_cols:
                    rec[f] = None  # field absent in model CSV
                    continue
                v = row.get(f, "")
                rec[f] = norm_bool(v) if f in BOOL_FIELDS else norm_cat(v)
            model[pid] = rec
    return model

def write_confusion_matrix_csv(out_path: Path, labels, cm):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["true\\pred"] + list(labels))
        for i, l in enumerate(labels):
            w.writerow([l] + list(cm[i]))

def main():
    ap = argparse.ArgumentParser(description="Evaluate model predictions vs. adjudicated gold.")
    ap.add_argument("--gold", required=True, help="Adjudicated gold CSV")
    ap.add_argument("--model", required=True, help="Model labels CSV from classifier")
    ap.add_argument("--outdir", default="reports/metrics", help="Where to save summary + confusion matrices")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    gold  = load_gold(Path(args.gold))
    model = load_model(Path(args.model))

    common = sorted(set(gold.keys()) & set(model.keys()))
    if not common:
        raise SystemExit("No overlapping post IDs between gold and model.")

    summary_path = outdir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as fp_sum:
        wsum = csv.DictWriter(fp_sum, fieldnames=[
            "field","n","accuracy","macro_precision","macro_recall","macro_f1","kappa","note"
        ])
        wsum.writeheader()

        print(f"Evaluating on {len(common)} overlapping posts\n")

        for field in EVAL_FIELDS:
            # Skip entire field if it's missing from the model CSV
            if all(model[pid].get(field) is None for pid in common):
                note = "skipped (field missing in model CSV)"
                print(f"{field:>24}: {note}")
                wsum.writerow({
                    "field": field, "n": 0,
                    "accuracy": "", "macro_precision": "", "macro_recall": "", "macro_f1": "", "kappa": "",
                    "note": note
                })
                continue

            y_true, y_pred = [], []
            for pid in common:
                gt = gold[pid].get(field, "")
                mp = model[pid].get(field, None)
                if mp is None or gt == "" or mp == "":
                    continue  # pairwise skip for blanks
                y_true.append(gt); y_pred.append(mp)

            n = len(y_true)
            if n == 0:
                note = "no overlapping non-empty labels"
                print(f"{field:>24}: {note}")
                wsum.writerow({
                    "field": field, "n": 0,
                    "accuracy": "", "macro_precision": "", "macro_recall": "", "macro_f1": "", "kappa": "",
                    "note": note
                })
                continue

            acc = accuracy_score(y_true, y_pred)
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )
            kappa = cohen_kappa_score(y_true, y_pred)

            print(f"{field:>24}:  n={n:4d}  acc={acc:.3f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  kappa={kappa:.3f}")
            # Optional detailed breakdown
            try:
                print(classification_report(y_true, y_pred, zero_division=0))
            except Exception:
                pass

            labels = sorted(set(y_true) | set(y_pred))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            write_confusion_matrix_csv(outdir / f"confmat_{field}.csv", labels, cm)

            wsum.writerow({
                "field": field, "n": n,
                "accuracy": f"{acc:.6f}",
                "macro_precision": f"{p:.6f}",
                "macro_recall": f"{r:.6f}",
                "macro_f1": f"{f1:.6f}",
                "kappa": f"{kappa:.6f}",
                "note": ""
            })

    print(f"\nSaved summary -> {summary_path}")
    print(f"Saved confusion matrices -> {outdir}/confmat_*.csv")

if __name__ == "__main__":
    main()
