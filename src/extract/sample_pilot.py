#!/usr/bin/env python3
"""
Purposive stratified pilot sampler for Phase 2 extraction.

The pilot exists to BREAK the schema, so a random draw (a third of it falls) is
the wrong design. Stratify on Phase 1's mechanism_of_injury: take up to
--per-stratum from every stratum (over-sampling rare mechanisms), then top up to
--n proportionally from the remaining posts, which favours the largest strata.
If the per-stratum floor alone exceeds --n, the floor is trimmed from the largest
strata first so rare mechanisms keep their full allocation.

--boost STRATUM=K raises one stratum's floor to K and exempts it from trimming —
used when a stratum-specific field needs enough cases to assess (e.g. fall_height
and landing_surface need falls, but 'fall' is the stratum trimmed first).

Seeded and reproducible. Methods must record that the pilot is PURPOSIVE, not random.
"""

import argparse, json, random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from common import find_repo_root, load_jsonl, newest_matching, get_labels, is_confirmed, require_selftext

STRATUM_KEY = "mechanism_of_injury"

def parse_boost(items) -> dict:
    out = {}
    for item in items or []:
        name, _, k = item.partition("=")
        if not k.isdigit():
            raise SystemExit(f"--boost expects STRATUM=K, got {item!r}")
        out[name] = int(k)
    return out

def allocate(sizes: dict, n: int, per_stratum: int, boost: dict = None) -> dict:
    boost = boost or {}
    alloc = {s: min(boost.get(s, per_stratum), k) for s, k in sizes.items()}
    by_size = sorted((s for s in sizes if s not in boost), key=lambda s: (-sizes[s], s))

    # Floor exceeds n: repeatedly trim the stratum with the highest allocation (larger
    # stratum first on ties), never below 1, so small strata keep their full floor longest
    if sum(alloc.values()) > n:
        print(f"[warn] per-stratum floor ({sum(alloc.values())}) exceeds --n {n}; trimming from largest strata")
        while sum(alloc.values()) > n:
            trimmable = [s for s in by_size if alloc[s] > 1]
            if not trimmable:
                break  # every stratum at 1 and still over n; keep one per stratum
            alloc[max(trimmable, key=lambda s: (alloc[s], sizes[s]))] -= 1
        return alloc

    # Top up proportionally to remaining stratum size (largest-remainder method)
    remaining = n - sum(alloc.values())
    avail = {s: sizes[s] - alloc[s] for s in sizes}
    pool = sum(avail.values())
    if remaining > 0 and pool > 0:
        remaining = min(remaining, pool)
        quotas = {s: remaining * avail[s] / pool for s in sizes}
        extra = {s: int(quotas[s]) for s in sizes}
        leftover = remaining - sum(extra.values())
        for s in sorted(sizes, key=lambda s: (-(quotas[s] - extra[s]), -sizes[s], s)):
            if leftover <= 0:
                break
            if extra[s] < avail[s]:
                extra[s] += 1
                leftover -= 1
        for s in sizes:
            alloc[s] += min(extra[s], avail[s])
    return alloc

def main():
    repo_root = find_repo_root(Path(__file__).resolve())
    interim = repo_root / "data" / "interim"

    ap = argparse.ArgumentParser(description="Purposive stratified pilot sample for Phase 2 extraction.")
    ap.add_argument("--in", dest="inp", default=None, help="Confirmed events JSONL; default: newest confirmed_events_n*.jsonl")
    ap.add_argument("--out", default=None, help="Output path; default: data/interim/pilot<N>_stratified.jsonl")
    ap.add_argument("--n", type=int, default=50, help="Pilot size")
    ap.add_argument("--per-stratum", type=int, default=4, help="Floor per mechanism stratum")
    ap.add_argument("--seed", type=int, default=2026, help="Random seed (reproducibility)")
    ap.add_argument("--boost", action="append", default=None, metavar="STRATUM=K",
                    help="Raise one stratum's floor to K, exempt from trimming (repeatable)")
    args = ap.parse_args()
    boost = parse_boost(args.boost)

    in_path = Path(args.inp).expanduser().resolve() if args.inp else newest_matching(interim, "confirmed_events_n*.jsonl")
    if not in_path.is_file():
        raise FileNotFoundError(f"Input not found: {in_path}")
    print(f"Using input: {in_path}")

    records = load_jsonl(in_path)
    require_selftext(records, in_path)
    confirmed = [r for r in records if is_confirmed(r)]
    if len(confirmed) != len(records):
        print(f"[warn] {len(records) - len(confirmed)} non-confirmed records in input ignored")

    strata = defaultdict(list)
    for r in confirmed:
        strata[get_labels(r).get(STRATUM_KEY) or "missing"].append(r)

    rng = random.Random(args.seed)
    for s in sorted(strata):
        strata[s].sort(key=lambda r: str(r.get("id")))
        rng.shuffle(strata[s])

    sizes = {s: len(v) for s, v in strata.items()}
    unknown = set(boost) - set(sizes)
    if unknown:
        raise SystemExit(f"--boost strata not present in input: {sorted(unknown)}")
    alloc = allocate(sizes, args.n, args.per_stratum, boost)

    print(f"\n{'stratum':<26}{'size':>7}{'pilot':>7}")
    for s in sorted(sizes, key=lambda s: (-sizes[s], s)):
        print(f"{s:<26}{sizes[s]:>7}{alloc[s]:>7}")
    total = sum(alloc.values())
    print(f"{'TOTAL':<26}{sum(sizes.values()):>7}{total:>7}")

    sample = []
    for s in sorted(sizes, key=lambda s: (-sizes[s], s)):
        sample.extend(strata[s][:alloc[s]])

    out_path = Path(args.out).expanduser().resolve() if args.out else interim / f"pilot{total}_stratified.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        for r in sample:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest_path = out_path.with_name(out_path.name.replace(".jsonl", "_manifest.json"))
    with open(manifest_path, "w", encoding="utf-8") as mfp:
        json.dump({
            "design": "purposive stratified (NOT random)",
            "stratum_key": f"phase1.{STRATUM_KEY}",
            "input_file": str(in_path),
            "output_file": str(out_path),
            "n_requested": args.n,
            "n_sampled": total,
            "per_stratum": args.per_stratum,
            "boost": boost,
            "seed": args.seed,
            "stratum_sizes": sizes,
            "pilot_composition": alloc,
            "ids": [r.get("id") for r in sample],
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, mfp, ensure_ascii=False, indent=2)

    print(f"\nWrote {total} pilot events -> {out_path}\nManifest -> {manifest_path}")

if __name__ == "__main__":
    main()
