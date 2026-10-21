#!/usr/bin/env python3
"""
Side-by-side comparison of Phase 2 extraction runs on the SAME posts (model choice).

Writes:
  agreement.csv    per field: how often runs populate it, and exact-value agreement
                   between each pair where both populated it
  review_sheet.csv one row per (post, field) with every run's value/status/span and
                   blank columns for the human reader's verdict
  posts.csv        post text for the review sheet (kept separate so text isn't repeated 24x)

Agreement between models is not accuracy — two models can agree on the same wrong
value. The review sheet is for the hand-reading that actually decides the schema.
"""

import argparse, csv, json
from itertools import combinations
from pathlib import Path

from common import find_repo_root, load_jsonl
from schema import FIELDS, field_type

def main():
    repo_root = find_repo_root(Path(__file__).resolve())

    ap = argparse.ArgumentParser(description="Compare Phase 2 extraction runs on the same posts.")
    ap.add_argument("--run", action="append", required=True, metavar="TAG=PATH",
                    help="An *_extractions.jsonl, labelled (repeat for each run, >= 2)")
    ap.add_argument("--outdir", default=None, help="Default: reports/extraction_qa/compare_<tags>/")
    args = ap.parse_args()

    runs = {}
    for item in args.run:
        tag, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--run expects TAG=PATH, got {item!r}")
        runs[tag] = {r["id"]: r for r in load_jsonl(Path(path).expanduser().resolve())}
    if len(runs) < 2:
        raise SystemExit("Need at least two --run inputs.")
    tags = list(runs)

    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    for tag in tags:
        extra = len(runs[tag]) - len(ids)
        if extra:
            print(f"[warn] {tag}: {extra} posts not in every run; ignored")
    print(f"Comparing {len(ids)} common posts across: {', '.join(tags)}")

    out_dir = Path(args.outdir).expanduser().resolve() if args.outdir else \
        repo_root / "reports" / "extraction_qa" / f"compare_{'_vs_'.join(tags)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def val(tag, i, f):
        v = runs[tag][i]["fields"][f]["value"]
        return "|".join(v) if isinstance(v, list) else v

    pairs = list(combinations(tags, 2))
    agree_rows = []
    print(f"\n{'field':<26}" + "".join(f"{('pop_' + t)[:12]:>13}" for t in tags)
          + "".join(f"{(a + '~' + b)[:16]:>17}" for a, b in pairs))
    for phase, cell, f in FIELDS:
        row = {"phase": phase, "cell": cell, "field": f, "type": field_type(f), "n_posts": len(ids)}
        line = f"{f:<26}"
        for t in tags:
            k = sum(val(t, i, f) is not None for i in ids)
            row[f"populated_{t}"] = k
            line += f"{k:>13}"
        for a, b in pairs:
            both = [(val(a, i, f), val(b, i, f)) for i in ids if val(a, i, f) is not None and val(b, i, f) is not None]
            same = sum(x == y for x, y in both)
            row[f"both_populated_{a}_{b}"] = len(both)
            # exact agreement only means something for enumerations; free text is left to the reader
            row[f"exact_agree_{a}_{b}"] = same if field_type(f) != "text" else ""
            line += f"{(f'{same}/{len(both)}' if field_type(f) != 'text' else f'-/{len(both)}'):>17}"
        agree_rows.append(row)
        print(line)

    with open(out_dir / "agreement.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(agree_rows[0].keys()))
        w.writeheader()
        w.writerows(agree_rows)

    sheet_fields = ["post_id", "title", "phase", "field"]
    for t in tags:
        sheet_fields += [f"{t}_value", f"{t}_status", f"{t}_span", f"{t}_violations"]
    sheet_fields += ["runs_agree", "human_best_run", "human_correct_value", "human_note"]
    with open(out_dir / "review_sheet.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=sheet_fields)
        w.writeheader()
        for i in ids:
            for phase, _, f in FIELDS:
                row = {"post_id": i, "title": runs[tags[0]][i]["title"], "phase": phase, "field": f}
                vals = []
                for t in tags:
                    c = runs[t][i]["fields"][f]
                    vals.append(val(t, i, f))
                    row[f"{t}_value"] = "" if vals[-1] is None else vals[-1]
                    row[f"{t}_status"] = c["status"]
                    row[f"{t}_span"] = c["evidence_span"] or ""
                    row[f"{t}_violations"] = "|".join(sorted({v["code"] for v in runs[t][i]["violations"] if v["field"] == f}))
                row["runs_agree"] = len(set(vals)) == 1
                w.writerow(row)

    with open(out_dir / "posts.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["post_id", "title", "selftext", "permalink", "phase1_mechanism_of_injury", "phase1_model_rationale"])
        for i in ids:
            r = runs[tags[0]][i]
            p1 = r.get("phase1_labels") or {}
            # phase1_model_rationale is the Phase 1 MODEL's note, not a human judgement
            w.writerow([i, r["title"], r.get("selftext", ""), r.get("permalink", ""),
                        p1.get("mechanism_of_injury", ""), p1.get("rationale_short", "")])

    print(f"\nWrote:\n  {out_dir / 'agreement.csv'}\n  {out_dir / 'review_sheet.csv'}\n  {out_dir / 'posts.csv'}")

if __name__ == "__main__":
    main()
