#!/usr/bin/env python3
import json, csv, random, gzip, math
from pathlib import Path
import argparse

# CSV columns for annotators
COLUMNS = [
    "post_id","title","selftext","permalink","created_utc","subreddit",
    "is_injury_event","mechanism_of_injury","nature_of_injury","body_region",
    "er_or_hospital_mentioned","age_group","rationale_short","coder_id"
]

def newest_raw_file(raw_dir: Path) -> Path:
    candidates = list(raw_dir.glob("*.jsonl")) + list(raw_dir.glob("*.jsonl.gz"))
    if not candidates:
        raise FileNotFoundError(f"No .jsonl/.jsonl.gz files found in {raw_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def open_maybe_gzip(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, "r", encoding="utf-8")

def stem_for_output(p: Path) -> str:
    name = p.name
    if name.endswith(".jsonl.gz"): return name[:-9]
    if name.endswith(".jsonl"):    return name[:-6]
    return p.stem

def main():
    ap = argparse.ArgumentParser(description="Sample 10% for annotation and write CSV + raw JSONL.GZ.")
    ap.add_argument("--in-jsonl", default=None, help="Optional explicit path; else newest in data/raw/")
    ap.add_argument("--out-csv", default="data/interim/gold/sample_for_annotation.csv", help="Output CSV path")
    ap.add_argument("--out-jsonl", default=None,
                    help="Optional output JSONL.GZ; default: data/interim/gold/<input_stem>_sample10pct_seed<SEED>.jsonl.gz")
    ap.add_argument("--seed", type=int, default=2025, help="Random seed (reproducibility)")
    args = ap.parse_args()

    # Locate input
    in_path = Path(args.in_jsonl).expanduser().resolve() if args.in_jsonl else newest_raw_file(Path("data/raw").resolve())
    if not in_path.is_file():
        raise FileNotFoundError(f"Input not found: {in_path}")
    print(f"Using input: {in_path}")

    # Load all posts
    posts = []
    with open_maybe_gzip(in_path) as f:
        for line in f:
            try:
                posts.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    N = len(posts)
    if N == 0:
        raise SystemExit("No posts found in input.")
    n_sample = max(1, math.ceil(0.10 * N))
    print(f"Total posts: {N} -> sampling 10% = {n_sample}")

    rng = random.Random(args.seed)
    sample = rng.sample(posts, n_sample)

    # Paths
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.out_jsonl:
        out_jsonl = Path(args.out_jsonl)
    else:
        base = stem_for_output(in_path)
        out_jsonl = Path("data/interim/gold") / f"{base}_sample10pct_seed{args.seed}.jsonl.gz"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV for annotators
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=COLUMNS)
        w.writeheader()
        for p in sample:
            w.writerow({
                "post_id": p.get("id"),
                "title": p.get("title"),
                "selftext": p.get("selftext"),
                "permalink": p.get("permalink"),
                "created_utc": p.get("created_utc"),
                "subreddit": p.get("subreddit"),
                "is_injury_event": "",
                "mechanism_of_injury": "",
                "nature_of_injury": "",
                "body_region": "",
                "er_or_hospital_mentioned": "",
                "age_group": "",
                "rationale_short": "",
                "coder_id": ""
            })

    # Write raw JSONL.GZ sample (exact original records)
    with gzip.open(out_jsonl, "wt", encoding="utf-8") as gz:
        for p in sample:
            gz.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Wrote {n_sample} rows to CSV: {out_csv}")
    print(f"Wrote {n_sample} raw posts to JSONL.GZ: {out_jsonl}")
    print("Move this JSONL.GZ to data/raw/ when you want the classifier to pick it up next.")

if __name__ == "__main__":
    main()
