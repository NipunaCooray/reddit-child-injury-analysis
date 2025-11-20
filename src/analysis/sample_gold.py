#!/usr/bin/env python3
import json, csv, random, gzip, math
from pathlib import Path
import argparse

# Define output CSV columns

COLUMNS = [
    "post_id","title","selftext","permalink","created_utc","subreddit",
    # annotation fields (blank for coders)
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

def main():
    ap = argparse.ArgumentParser(description="Sample 10% of posts for human annotation (gold standard).")
    ap.add_argument("--in-jsonl", default=None, help="Optional explicit path. If omitted, use newest in data/raw/")
    ap.add_argument("--out-csv", default="data/interim/gold/sample_for_annotation.csv", help="Output CSV path")
    ap.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    args = ap.parse_args()

    # Locate input
    if args.in_jsonl:
        in_path = Path(args.in_jsonl).expanduser().resolve()
        if not in_path.is_file():
            raise FileNotFoundError(f"Input not found: {in_path}")
    else:
        in_path = newest_raw_file(Path("data/raw").resolve())
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
    n_sample = max(1, math.ceil(0.10 * N))  # 10% ceiling, at least 1
    print(f"Total posts: {N} -> sampling 10% = {n_sample}")

    rng = random.Random(args.seed)
    sample = rng.sample(posts, n_sample)

    # Write CSV
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)


    with open(out_path, "w", newline="", encoding="utf-8") as fp:
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

    print(f"Wrote {n_sample} rows to {out_path}")

if __name__ == "__main__":
    main()
