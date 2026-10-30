#!/usr/bin/env python3
"""
Export the codebook and a human-coding sample as the JSON documents the coding app
(a claude.ai artifact) reads from its database.

  codebook.json            -> config/codebook   (fields, values, notes, rules; from schema.py)
  tutorial/<id>.json       -> tutorial/<id>     (constructed practice posts + answer key; from tutorial.json)
  posts/<sample>/<id>.json -> posts/<id>        (title, body, permalink only)

The posts carry NO model output and NO Phase 1 labels or rationale, so the coder
stays blind. The files hold full post text, so they are written under reports/
(git-ignored) and loaded into the app with ArtifactData batch writes (50 per batch).
The tutorial posts are invented, so their source lives in the repo (tutorial.json); the
app reveals each key only after the person has coded that post.
"""

import argparse, json
from pathlib import Path

from common import find_repo_root, load_jsonl, require_selftext
from schema import (FIELDS, VOCAB, DEFINITIONS, VALUE_NOTES, PHASE_TITLES, SCHEMA_VERSION, STATUSES,
                    FREE_TEXT_MAX, AUDIT_FIELD, FALL_MECHANISMS, CONDITIONAL_FALL_FIELDS, field_type)

STATUS_HELP = {
    "stated": "The post says it. The quote is copied word for word from the post.",
    "inferred": "The post implies it. The quote is the text that carries the inference.",
    "not_stated": "The post gives no support. No value, no quote.",
}

def codebook() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "statuses": list(STATUSES),
        "status_help": STATUS_HELP,
        "free_text_max": FREE_TEXT_MAX,
        "phases": [{"id": p, "title": PHASE_TITLES[p]} for p in dict.fromkeys(p for p, _, _ in FIELDS)],
        "fields": [{
            "phase": p, "cell": c, "field": f, "type": field_type(f),
            "definition": DEFINITIONS[f],
            "values": [{"value": v, "note": VALUE_NOTES.get(f, {}).get(v, "")} for v in VOCAB.get(f, ())],
        } for p, c, f in FIELDS],
        "rules": {
            "audit_field": AUDIT_FIELD,  # never not_stated; use 'unclear'
            "fall_mechanisms": sorted(FALL_MECHANISMS),
            "conditional_fall_fields": list(CONDITIONAL_FALL_FIELDS),  # not_stated unless mechanism_iceci is a fall
        },
    }

def main():
    repo_root = find_repo_root(Path(__file__).resolve())

    ap = argparse.ArgumentParser(description="Export codebook and a coding sample for the coding app.")
    ap.add_argument("--posts", default=None, help="Sample JSONL (e.g. data/interim/deeptier120_stratified.jsonl)")
    ap.add_argument("--sample", default=None, help="Sample name shown in the app (e.g. practice, deeptier)")
    ap.add_argument("--outdir", default=None, help="Default: reports/coding_app/")
    args = ap.parse_args()
    if bool(args.posts) != bool(args.sample):
        raise SystemExit("--posts and --sample go together.")

    out_dir = Path(args.outdir).expanduser().resolve() if args.outdir else repo_root / "reports" / "coding_app"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "codebook.json").write_text(json.dumps(codebook(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote codebook (schema {SCHEMA_VERSION}) -> {out_dir / 'codebook.json'}")

    tut = json.loads((Path(__file__).resolve().parent / "tutorial.json").read_text(encoding="utf-8"))
    if tut["schema_version"] != SCHEMA_VERSION:
        raise SystemExit(f"tutorial.json is keyed to schema {tut['schema_version']}, not {SCHEMA_VERSION}: "
                         "re-check every key against the new definitions, then update its schema_version.")
    tut_dir = out_dir / "tutorial"
    tut_dir.mkdir(parents=True, exist_ok=True)
    for p in tut["posts"]:
        doc = {k: p[k] for k in ("id", "order", "label", "title", "selftext", "takeaways")}
        doc.update(sample="tutorial", schema_version=SCHEMA_VERSION,
                   key={f: {"value": c["value"], "status": c["status"], "quote": c["evidence_span"] or "", "why": c["why"]}
                        for f, c in p["key"].items()})
        (tut_dir / f"{p['id']}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(tut['posts'])} tutorial posts -> {tut_dir}")

    if args.posts:
        in_path = Path(args.posts).expanduser().resolve()
        records = load_jsonl(in_path)
        require_selftext(records, in_path)
        post_dir = out_dir / "posts" / args.sample
        post_dir.mkdir(parents=True, exist_ok=True)
        for n, r in enumerate(records, 1):
            doc = {"id": r["id"], "title": r.get("title") or "", "selftext": r.get("selftext") or "",
                   "permalink": r.get("permalink") or "", "sample": args.sample, "order": n}
            (post_dir / f"{r['id']}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(records)} posts ({args.sample}) -> {post_dir}")

if __name__ == "__main__":
    main()
