#!/usr/bin/env python3
"""
Write the Phase 2 analytic dataset: confirmed injury events only.

Phase 1 writes ALL posts (kept and rejected) to *_labels.jsonl. This filters to
is_injury_event == true, asserts the count against Phase 1's *_timing.json 'kept'
value (and --expect if given), and bakes the count into the output filename.
"""

import argparse, json
from pathlib import Path

from common import find_repo_root, load_jsonl, newest_matching, is_confirmed, require_selftext

def main():
    repo_root = find_repo_root(Path(__file__).resolve())
    interim = repo_root / "data" / "interim"

    ap = argparse.ArgumentParser(description="Filter Phase 1 labels JSONL to confirmed injury events.")
    ap.add_argument("--in", dest="inp", default=None, help="Phase 1 *_labels.jsonl; default: newest in data/interim/")
    ap.add_argument("--out", default=None, help="Output path; default: data/interim/confirmed_events_n<N>.jsonl")
    ap.add_argument("--expect", type=int, default=None, help="Assert the confirmed count equals N")
    args = ap.parse_args()

    in_path = Path(args.inp).expanduser().resolve() if args.inp else newest_matching(interim, "*_labels.jsonl")
    if not in_path.is_file():
        raise FileNotFoundError(f"Input not found: {in_path}")
    print(f"Using input: {in_path}")

    records = load_jsonl(in_path)
    require_selftext(records, in_path)
    confirmed = [r for r in records if is_confirmed(r)]
    n = len(confirmed)
    print(f"Read {len(records)} records -> {n} confirmed injury events ({len(records) - n} rejected)")

    # Assert against Phase 1's own kept counter
    timing_path = in_path.with_name(in_path.name.replace("_labels.jsonl", "_timing.json"))
    if timing_path.is_file():
        kept = json.loads(timing_path.read_text(encoding="utf-8")).get("kept")
        print(f"Phase 1 timing ({timing_path.name}): kept={kept}")
        if kept != n:
            raise SystemExit(f"COUNT MISMATCH: filtered {n} but Phase 1 timing says kept={kept}. Resolve before running extraction.")
    elif args.expect is None:
        raise SystemExit(f"Phase 1 timing file not found ({timing_path}); pass --expect N to assert the count manually.")
    else:
        print(f"[warn] Phase 1 timing file not found ({timing_path}); asserting against --expect only")

    if args.expect is not None and args.expect != n:
        raise SystemExit(f"COUNT MISMATCH: filtered {n} but --expect {args.expect}.")

    out_path = Path(args.out).expanduser().resolve() if args.out else interim / f"confirmed_events_n{n}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        for r in confirmed:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {n} confirmed events -> {out_path}")

if __name__ == "__main__":
    main()
