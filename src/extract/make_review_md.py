#!/usr/bin/env python3
"""
Write a single Markdown file for a light-touch hand review of one extraction run:
a definitions reference (from schema.py), then each post followed by the model's cells.
Field names link to their definitions. Contains full post text, so it is written under
reports/ (git-ignored) and should stay local.
"""

import argparse
from pathlib import Path

from common import find_repo_root, load_jsonl
from schema import (FIELDS, VOCAB, DEFINITIONS, VALUE_NOTES, PHASE_TITLES, SCHEMA_VERSION,
                    field_type)

STATUS_HELP = {
    "stated": "the post says it; the quote is verbatim from the post",
    "inferred": "the post implies it; the quote is the text that carries the inference",
    "not_stated": "the post gives no support; no value, no quote",
}

def cell_text(s):
    return " ".join(str(s).split()).replace("|", "\\|")

def val_str(v):
    if v is None:
        return ""
    return ", ".join(v) if isinstance(v, list) else str(v)

def main():
    repo_root = find_repo_root(Path(__file__).resolve())

    ap = argparse.ArgumentParser(description="Markdown hand-review file for one extraction run.")
    ap.add_argument("--in", dest="inp", required=True, help="*_extractions.jsonl")
    ap.add_argument("--out", default=None, help="Default: reports/extraction_qa/hand_review/<input stem>_review.md")
    args = ap.parse_args()

    in_path = Path(args.inp).expanduser().resolve()
    records = load_jsonl(in_path)
    model = records[0].get("model") if records else "?"
    versions = sorted({r.get("extraction_schema_version") for r in records if r.get("extraction_schema_version")})
    out_path = Path(args.out).expanduser().resolve() if args.out else \
        repo_root / "reports" / "extraction_qa" / "hand_review" / f"{in_path.stem}_review.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    L = [f"# Hand review: {model}", "",
         f"{len(records)} posts · extraction schema {', '.join(versions)} · definitions below are from schema {SCHEMA_VERSION}", "",
         "Local file: it contains full post text. Do not commit, email or upload it.", "",
         "**How to use:** read each post, then check the model's answers under it. Field names link to their definitions. "
         "Note anything wrong under **My notes** for that post.", "",
         "Jump to: [Definitions](#definitions) · " + " · ".join(f"[{n}](#post-{n})" for n in range(1, len(records) + 1, 10)), "",
         "## Definitions", "",
         "**Status**", ""]
    L += [f"- `{s}` — {h}" for s, h in STATUS_HELP.items()]
    L += ["", "Prefer `not_stated` over a guess: an inference must rest on words in *this* post, "
              "not on what parenting posts typically say.", ""]
    phase = None
    for p, cell, f in FIELDS:
        if p != phase:
            L += [f"### {PHASE_TITLES[p].title()}", ""]
            phase = p
        L += [f'<a id="{f}"></a>', f"#### {f}", "", f"*{cell.replace('_', ' ')} · {field_type(f).replace('_', ' ')}*", "",
              DEFINITIONS[f], ""]
        if f in VOCAB:
            notes = VALUE_NOTES.get(f, {})
            L += [f"- `{v}`" + (f" — {notes[v]}" if v in notes else "") for v in VOCAB[f]]
            L.append("")

    for n, r in enumerate(records, 1):
        p1 = (r.get("phase1_labels") or {}).get("mechanism_of_injury", "")
        L += ["---", "", f'<a id="post-{n}"></a>', f"## Post {n}: {cell_text(r['title'])}", "",
              f"`{r['id']}` · Phase 1 mechanism: {p1} · [permalink](https://www.reddit.com{r.get('permalink', '')})", ""]
        body = (r.get("selftext") or "").strip() or "(no body)"
        L += ["> " + line if line.strip() else ">" for line in body.split("\n")]
        L += ["", "| Field | Value | Status | Quote |", "| --- | --- | --- | --- |"]
        blank = []
        for _, _, f in FIELDS:
            c = r["fields"][f]
            if c["status"] == "not_stated":
                blank.append(f"[{f}](#{f})")
                continue
            L.append(f"| [{f}](#{f}) | {cell_text(val_str(c['value']))} | {c['status']} | {cell_text(c['evidence_span'] or '')} |")
        L += ["", "**Not stated:** " + (", ".join(blank) if blank else "none"), ""]
        flags = sorted({f"{v['field']}: {v['code']}" for v in r.get("violations", [])})
        if flags:
            L += ["**Guardrail flags:** " + "; ".join(flags), ""]
        L += ["**My notes:**", "", ""]

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {out_path} ({len(records)} posts)")

if __name__ == "__main__":
    main()
