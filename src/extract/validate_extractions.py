#!/usr/bin/env python3
"""
Automated QA tier for Phase 2 extractions.

Reads an *_extractions.jsonl and reports grounding, vocabulary and coherence
violation rates plus per-field stated/inferred/not_stated coverage. Flags any
field above 80% not_stated for redefinition or cutting BEFORE the full run —
except the scope-limited fields (originating post only; see README), and with
the fall-conditional fields judged over fall events only.

The numbers only tell you where to look. Read every pilot output by hand.
"""

import argparse, csv, json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from common import find_repo_root, load_jsonl
from schema import (FIELDS, FIELD_NAMES, FALL_MECHANISMS, CONDITIONAL_FALL_FIELDS,
                    AUDIT_FIELD, AUDIT_NEGATIVE, AUDIT_NEAR_MISS,
                    SCOPE_LIMITED_FIELDS, RARE_BY_DESIGN_FIELDS, STATUSES, field_type)

NOT_STATED_CUT = 0.80
MIN_N_FOR_CUT = 10                   # below this the not_stated rate is too noisy to act on
WATCH_FIELD = "supervision_status"   # the field the prevention argument leans on hardest

def pct(k, n):
    return (100.0 * k / n) if n else float("nan")

def main():
    repo_root = find_repo_root(Path(__file__).resolve())

    ap = argparse.ArgumentParser(description="QA report for Phase 2 extractions.")
    ap.add_argument("--in", dest="inp", required=True, help="*_extractions.jsonl")
    ap.add_argument("--outdir", default=None, help="Default: reports/extraction_qa/")
    args = ap.parse_args()

    in_path = Path(args.inp).expanduser().resolve()
    out_dir = Path(args.outdir).expanduser().resolve() if args.outdir else repo_root / "reports" / "extraction_qa"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(in_path)
    if not records or "fields" not in records[0]:
        raise SystemExit(f"{in_path} does not look like an extractor output (no 'fields').")
    n_events = len(records)
    parsed = [r for r in records if r.get("parse_ok", True)]
    n_parsed = len(parsed)
    print(f"Using input: {in_path}")
    print(f"Events: {n_events} | parsed: {n_parsed} | parser errors: {n_events - n_parsed}")

    # ---- coverage (parsed events only; parser errors would inflate not_stated)
    fall_events = [r for r in parsed if r["fields"]["mechanism_iceci"]["value"] in FALL_MECHANISMS]
    fall_ids = {id(r) for r in fall_events}
    status_counts = {f: Counter() for f in FIELD_NAMES}
    for r in parsed:
        for f in FIELD_NAMES:
            if f in CONDITIONAL_FALL_FIELDS and id(r) not in fall_ids:
                continue
            status_counts[f][r["fields"][f]["status"]] += 1

    # ---- violations: count distinct (event, field, code) cells
    cell_codes = Counter()          # code -> n cells
    field_codes = Counter()         # (field, code) -> n events
    events_with_any = 0
    events_with_code = Counter()
    for r in records:
        seen = set()
        for v in r.get("violations", []):
            seen.add((v.get("field"), v["code"]))
        events_with_any += bool(seen)
        for field, code in seen:
            cell_codes[code] += 1
            field_codes[(field, code)] += 1
        for code in {c for _, c in seen}:
            events_with_code[code] += 1

    final_stated = sum(status_counts[f]["stated"] for f in FIELD_NAMES)
    originally_stated = final_stated + cell_codes["stated_without_span"] + cell_codes["unsupported_span"]
    populated_inferred = sum(status_counts[f]["inferred"] for f in FIELD_NAMES)
    enum_fields = [f for f in FIELD_NAMES if field_type(f) in ("enum", "multi_enum")]
    populated_enum = sum(n_parsed - status_counts[f]["not_stated"] for f in enum_fields)

    rates = {
        "unsupported_span_rate": pct(cell_codes["unsupported_span"], originally_stated),
        "stated_without_span_rate": pct(cell_codes["stated_without_span"], originally_stated),
        "inferred_span_unsupported_rate": pct(cell_codes["inferred_span_unsupported"],
                                              populated_inferred - cell_codes["stated_without_span"] - cell_codes["unsupported_span"]),
        "enum_violation_rate": pct(cell_codes["enum_violation"], populated_enum),
        "enum_ambiguous_rate": pct(cell_codes["enum_ambiguous"], populated_enum),
        "enum_snapped_rate": pct(cell_codes["enum_snapped"], populated_enum),
        "conditional_violation_event_rate": pct(events_with_code["conditional_violation"], n_events),
        "category_without_object_event_rate": pct(events_with_code["category_without_object"], n_events),
        "malformed_cell_rate": pct(cell_codes["malformed_cell"], n_parsed * len(FIELD_NAMES)),
        "parser_error_rate": pct(n_events - n_parsed, n_events),
        "events_with_any_violation_rate": pct(events_with_any, n_events),
    }

    # ---- console report
    print("\nGrounding (denominator: cells the model claimed 'stated' = %d)" % originally_stated)
    print(f"  unsupported span          : {cell_codes['unsupported_span']:>5}  ({rates['unsupported_span_rate']:.1f}%)")
    print(f"  stated without span       : {cell_codes['stated_without_span']:>5}  ({rates['stated_without_span_rate']:.1f}%)")
    print(f"  inferred span not in post : {cell_codes['inferred_span_unsupported']:>5}  ({rates['inferred_span_unsupported_rate']:.1f}% of inferred cells)")
    print(f"  elided span (accepted)    : {cell_codes['elided_span']:>5}")
    print("Vocabulary (denominator: populated enum cells = %d)" % populated_enum)
    print(f"  enum violation            : {cell_codes['enum_violation']:>5}  ({rates['enum_violation_rate']:.1f}%)")
    print(f"  enum ambiguous            : {cell_codes['enum_ambiguous']:>5}  ({rates['enum_ambiguous_rate']:.1f}%)")
    print(f"  enum snapped              : {cell_codes['enum_snapped']:>5}  ({rates['enum_snapped_rate']:.1f}%)")
    print("Coherence / structure")
    print(f"  conditional violations    : {cell_codes['conditional_violation']:>5}  (events: {rates['conditional_violation_event_rate']:.1f}%)")
    print(f"  category without object   : {cell_codes['category_without_object']:>5}")
    print(f"  malformed cells           : {cell_codes['malformed_cell']:>5}  ({rates['malformed_cell_rate']:.1f}% of cells)")
    print(f"  bad status                : {cell_codes['bad_status']:>5}")
    print(f"  keys recovered            : {cell_codes['key_recovered']:>5}")
    print(f"  parser errors             : {n_events - n_parsed:>5}  ({rates['parser_error_rate']:.1f}%)")
    print(f"  events with >=1 violation : {events_with_any:>5}  ({rates['events_with_any_violation_rate']:.1f}%)")

    coverage_rows, flags, scope_notes, rare_notes, small_n = [], [], [], [], []
    print(f"\n{'field':<26}{'n':>5}{'stated':>9}{'inferred':>10}{'not_st':>9}  flag")
    for phase, cell, f in FIELDS:
        c = status_counts[f]
        n = sum(c.values())
        ns_rate = (c["not_stated"] / n) if n else float("nan")
        flag = ""
        if n and ns_rate > NOT_STATED_CUT:
            if f in SCOPE_LIMITED_FIELDS:
                flag = "scope_limited"
                scope_notes.append(f)
            elif f in RARE_BY_DESIGN_FIELDS:
                flag = "rare_by_design"
                rare_notes.append(f)
            elif n < MIN_N_FOR_CUT:
                flag = "insufficient_n"
                small_n.append(f)
            else:
                flag = "REDEFINE_OR_CUT"
                flags.append(f)
        if f == WATCH_FIELD:
            flag = (flag + " WATCH").strip()
        denom_note = " (fall events)" if f in CONDITIONAL_FALL_FIELDS else ""
        print(f"{f:<26}{n:>5}{pct(c['stated'], n):>8.0f}%{pct(c['inferred'], n):>9.0f}%{pct(c['not_stated'], n):>8.0f}%  {flag}{denom_note}")
        coverage_rows.append({
            "phase": phase, "cell": cell, "field": f, "type": field_type(f),
            "denominator": "fall_events" if f in CONDITIONAL_FALL_FIELDS else "parsed_events",
            "n": n, **{s: c[s] for s in STATUSES},
            **{f"pct_{s}": round(pct(c[s], n), 1) for s in STATUSES},
            "flag": flag,
        })

    if not fall_events:
        print("\n[note] no fall events in this file; fall_height/landing_surface coverage not assessable")
    for f in flags:
        print(f"\n[FLAG] {f}: {pct(status_counts[f]['not_stated'], sum(status_counts[f].values())):.0f}% not_stated "
              f"(> {NOT_STATED_CUT:.0%}). Redefine or cut this field BEFORE the full run.")
    for f in rare_notes:
        print(f"\n[rare] {f}: high not_stated is expected (most posts carry no such statement). "
              f"Judge on whether the captured values are correct, not on coverage.")
    for f in small_n:
        print(f"\n[small n] {f}: only {sum(status_counts[f].values())} applicable events (< {MIN_N_FOR_CUT}); "
              f"coverage not assessable — enlarge the pilot for this field.")
    for f in scope_notes:
        print(f"\n[scope] {f}: high not_stated is expected (originating post only; comment threads not collected). "
              f"Do not cut mechanically — judge whether the captured values are correct.")
    w = status_counts[WATCH_FIELD]
    print(f"\n[watch] {WATCH_FIELD}: {pct(w['stated'], sum(w.values())):.0f}% stated, "
          f"{pct(w['inferred'], sum(w.values())):.0f}% inferred, {pct(w['not_stated'], sum(w.values())):.0f}% not_stated. "
          "Hand-check every inferred value.")

    # ---- Phase 1 audit: the extraction model's independent injury_event_check vs Phase 1
    # Every input post was accepted by Phase 1, so not_an_injury_event marks a candidate
    # false positive. Candidates, not errors: each needs a human read. Near misses are
    # listed too but counted separately: whether they are in scope is a study decision.
    audit = {"available": AUDIT_FIELD in parsed[0]["fields"]} if parsed else {"available": False}
    disagreements = []
    if audit["available"]:
        values = Counter(r["fields"][AUDIT_FIELD]["value"] or "not_stated" for r in parsed)
        for r in parsed:
            v = r["fields"][AUDIT_FIELD]["value"]
            if v in AUDIT_NEGATIVE | AUDIT_NEAR_MISS:
                disagreements.append({"id": r["id"], "title": r["title"], "injury_event_check": v, "status": r["fields"][AUDIT_FIELD]["status"],
                                      "evidence_span": r["fields"][AUDIT_FIELD]["evidence_span"] or "",
                                      "permalink": r.get("permalink", "")})
        n_fp = sum(d["injury_event_check"] in AUDIT_NEGATIVE for d in disagreements)
        n_nm = len(disagreements) - n_fp
        audit.update({"values": dict(values), "n": n_parsed, "n_candidate_false_positives": n_fp,
                      "pct_candidate_false_positives": round(pct(n_fp, n_parsed), 1),
                      "n_near_miss": n_nm, "pct_near_miss": round(pct(n_nm, n_parsed), 1)})
        print(f"\nPhase 1 audit ({AUDIT_FIELD})")
        for v, k in values.most_common():
            print(f"  {v:<26}{k:>5}  ({pct(k, n_parsed):.0f}%)")
        print(f"  -> candidate false positives: {n_fp}/{n_parsed} ({pct(n_fp, n_parsed):.0f}%); "
              f"near misses: {n_nm}/{n_parsed} ({pct(n_nm, n_parsed):.0f}%). "
              "Hand-check each in audit_disagreements.csv before citing a rate.")
        for d in disagreements:
            print(f"     {d['id']}  {d['injury_event_check']:<22} {d['title'][:60]}")

    # ---- write outputs
    cov_path = out_dir / "field_coverage.csv"
    with open(cov_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(coverage_rows[0].keys()))
        writer.writeheader()
        writer.writerows(coverage_rows)

    viol_path = out_dir / "violations_by_field.csv"
    with open(viol_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["field", "code", "n_events", "pct_events"])
        for (field, code), k in sorted(field_codes.items(), key=lambda x: (-x[1], str(x[0][0]), x[0][1])):
            writer.writerow([field or "", code, k, round(pct(k, n_events), 1)])

    audit_path = None
    if audit["available"]:
        audit_path = out_dir / "audit_disagreements.csv"
        with open(audit_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["id", "title", "injury_event_check",
                                                    "status", "evidence_span", "permalink", "human_verdict"])
            writer.writeheader()
            writer.writerows(disagreements)

    summary_path = out_dir / "qa_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump({
            "input_file": str(in_path),
            "models": dict(Counter(r.get("model") for r in records)),
            "extraction_schema_versions": dict(Counter(r.get("extraction_schema_version") for r in records)),
            "n_events": n_events,
            "n_parsed": n_parsed,
            "n_fall_events": len(fall_events),
            "originally_stated_cells": originally_stated,
            "populated_enum_cells": populated_enum,
            "violation_cells_by_code": dict(cell_codes.most_common()),
            "events_by_code": dict(events_with_code.most_common()),
            "rates_pct": {k: (None if v != v else round(v, 2)) for k, v in rates.items()},
            "not_stated_cut": NOT_STATED_CUT,
            "flag_redefine_or_cut": flags,
            "scope_limited_above_cut": scope_notes,
            "rare_by_design_above_cut": rare_notes,
            "insufficient_n_above_cut": small_n,
            "phase1_audit": audit,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, fp, ensure_ascii=False, indent=2)

    print(f"\nWrote:\n  {cov_path}\n  {viol_path}\n  {summary_path}" + (f"\n  {audit_path}" if audit_path else ""))

if __name__ == "__main__":
    main()
