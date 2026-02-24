#!/usr/bin/env python3
"""
Summarize classified Reddit injury labels (the full dataset)

Reads the newest *_labels.csv in data/interim/ (or a path you pass),
then writes summary CSVs + a Markdown summary to reports/summaries/.

Outputs:
- summary_overall.csv                  # totals and headline rates
- counts_by_field_all.csv              # value counts for each field (all rows)
- counts_by_field_injury_only.csv      # value counts for each field (is_injury_event==True)
- crosstab_mechanism_by_age.csv
- crosstab_nature_by_body_region.csv
- timeseries_injury_events_daily.csv   # per-day counts of injury events
- top_subreddits.csv                   # subreddits ranked by injury posts
- SUMMARY.md                           # human-readable summary

Usage:
  python src/analysis/summarize_labels.py
  python src/analysis/summarize_labels.py --in-csv data/interim/<your_file>_labels.csv
"""

import argparse
from pathlib import Path
import csv
from datetime import datetime, timezone

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("This script requires pandas. Try:  pip install pandas") from e

# Fields written by your classifier
EXPECTED = [
    "id","subreddit","created_utc","permalink","title",
    "is_injury_event","mechanism_of_injury","nature_of_injury","body_region",
    "er_or_hospital_mentioned","age_group","rationale_short"
]

INTERIM_DIR = Path("data/interim")
REPORT_DIR  = Path("reports/summaries")

def newest_labels_file(interim: Path) -> Path:
    candidates = sorted(interim.glob("*_labels.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No *_labels.csv found in {interim}")
    return candidates[0]

def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # normalize basic types
    # booleans may be 'true/false' or 'True/False' or blanks → map to True/False
    def _to_bool(s: str) -> bool:
        s = (s or "").strip().lower()
        if s in {"true","1","yes","y"}:  return True
        if s in {"false","0","no","n"}:  return False
        return False  # safe default
    df["is_injury_event"] = df.get("is_injury_event", "").apply(_to_bool)
    df["er_or_hospital_mentioned"] = df.get("er_or_hospital_mentioned", "").apply(_to_bool)

    # created_utc may be epoch seconds (int-like) or ISO string; try both
    def _to_date(x):
        s = str(x).strip()
        if s.isdigit():
            try:
                dt = datetime.fromtimestamp(int(s), tz=timezone.utc)
                return dt.date()
            except Exception:
                return None
        # try ISO
        try:
            dt = datetime.fromisoformat(s.replace("Z","+00:00"))
            return dt.date()
        except Exception:
            return None
    df["created_date_utc"] = df.get("created_utc", "").apply(_to_date)

    return df

def write_csv(path: Path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

def summarize(df: pd.DataFrame, outdir: Path, input_path: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    N = len(df)
    n_injury = int(df["is_injury_event"].sum())
    p_injury = (n_injury / N * 100.0) if N else 0.0
    n_er_all = int(df["er_or_hospital_mentioned"].sum())
    # Among injury-only
    df_inj = df[df["is_injury_event"] == True].copy()
    n_er_inj = int(df_inj["er_or_hospital_mentioned"].sum())
    p_er_inj = (n_er_inj / len(df_inj) * 100.0) if len(df_inj) else 0.0

    # Overall summary table
    write_csv(
        outdir / "summary_overall.csv",
        rows=[
            ["input_file", str(input_path)],
            ["total_rows", N],
            ["injury_events_n", n_injury],
            ["injury_events_pct", f"{p_injury:.2f}"],
            ["er_mentioned_all_n", n_er_all],
            ["er_mentioned_in_injury_n", n_er_inj],
            ["er_mentioned_in_injury_pct", f"{p_er_inj:.2f}"],
        ],
        header=["metric","value"]
    )

    # Value counts ALL rows (useful to sanity-check enums)
    fields_for_counts = [
        "mechanism_of_injury","nature_of_injury","body_region",
        "age_group","er_or_hospital_mentioned","is_injury_event","subreddit"
    ]
    rows = []
    for col in fields_for_counts:
        vc = df[col].astype(str).str.strip().str.lower().value_counts(dropna=False)
        for val, cnt in vc.items():
            rows.append([col, val, int(cnt)])
    write_csv(outdir / "counts_by_field_all.csv", rows, header=["field","value","count"])

    # Value counts INJURY ONLY (drop NA for created fields)
    rows_inj = []
    for col in ["mechanism_of_injury","nature_of_injury","body_region","age_group","er_or_hospital_mentioned","subreddit"]:
        vc = df_inj[col].astype(str).str.strip().str.lower().value_counts(dropna=False)
        for val, cnt in vc.items():
            rows_inj.append([col, val, int(cnt)])
    write_csv(outdir / "counts_by_field_injury_only.csv", rows_inj, header=["field","value","count"])

    # Cross-tabs
    ct_mech_age = pd.crosstab(
        df_inj["mechanism_of_injury"].str.lower(),
        df_inj["age_group"].str.lower(),
        dropna=False
    ).sort_index()
    ct_mech_age.to_csv(outdir / "crosstab_mechanism_by_age.csv")

    ct_nat_body = pd.crosstab(
        df_inj["nature_of_injury"].str.lower(),
        df_inj["body_region"].str.lower(),
        dropna=False
    ).sort_index()
    ct_nat_body.to_csv(outdir / "crosstab_nature_by_body_region.csv")

    # Time series (injury events per day, UTC)
    ts = (
        df[df["is_injury_event"] & df["created_date_utc"].notna()]
        .groupby("created_date_utc", as_index=False)
        .size()
        .rename(columns={"size":"injury_events"})
        .sort_values("created_date_utc")
    )
    ts.to_csv(outdir / "timeseries_injury_events_daily.csv", index=False)

    # Top subreddits by injury posts
    top_subs = (
        df_inj["subreddit"].str.lower().value_counts()
        .rename_axis("subreddit").reset_index(name="injury_posts")
        .sort_values("injury_posts", ascending=False)
    )
    top_subs.to_csv(outdir / "top_subreddits.csv", index=False)

    # Markdown summary
    md = outdir / "SUMMARY.md"
    with open(md, "w", encoding="utf-8") as fp:
        fp.write("# Classification Summary\n\n")
        fp.write(f"- **Input file:** `{input_path}`\n")
        fp.write(f"- **Total posts:** {N}\n")
        fp.write(f"- **Injury events:** {n_injury} ({p_injury:.2f}%)\n")
        fp.write(f"- **ER mentioned (injury-only):** {n_er_inj} ({p_er_inj:.2f}%)\n")
        if not ts.empty:
            first = ts.iloc[0]["created_date_utc"]
            last  = ts.iloc[-1]["created_date_utc"]
            fp.write(f"- **Date span (UTC):** {first} → {last}\n")
        fp.write("\n## Files written\n")
        for p in [
            "summary_overall.csv",
            "counts_by_field_all.csv",
            "counts_by_field_injury_only.csv",
            "crosstab_mechanism_by_age.csv",
            "crosstab_nature_by_body_region.csv",
            "timeseries_injury_events_daily.csv",
            "top_subreddits.csv",
        ]:
            fp.write(f"- `{p}`\n")

    # Console echo
    print("Summary written to:", outdir)
    print(f"  total={N} | injury_n={n_injury} ({p_injury:.2f}%) | ER_in_injury={n_er_inj} ({p_er_inj:.2f}%)")
    if not ts.empty:
        print(f"  timeseries rows: {len(ts)}  ({ts['created_date_utc'].iloc[0]} → {ts['created_date_utc'].iloc[-1]})")

def parse_args():
    ap = argparse.ArgumentParser(description="Summarize injury classification outputs.")
    ap.add_argument("--in-csv", default=None, help="Path to *_labels.csv (default: newest in data/interim/)")
    return ap.parse_args()

def main():
    args = parse_args()
    labels_path = Path(args.in_csv).resolve() if args.in_csv else newest_labels_file(INTERIM_DIR)
    print("Using:", labels_path)
    df = load_labels(labels_path)
    summarize(df, REPORT_DIR, labels_path)

if __name__ == "__main__":
    main()
