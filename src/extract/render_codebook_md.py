#!/usr/bin/env python3
"""
Render the Phase 2 codebook from schema.py as Markdown, for reading and sharing.

  python src/extract/render_codebook_md.py            # -> reports/codebook_<version>.md

schema.py stays the source of truth; regenerate after every version bump.
"""

import argparse
from pathlib import Path

from common import find_repo_root
import schema as s
from export_coding_app import STATUS_HELP

HEAD = {"pre_event": "Pre-event", "event": "Event", "post_event": "Post-event",
        "interpretive": "Interpretive (outside Haddon)", "audit": "Phase 1 audit (independent judgement)"}
CELL = {"host": "Host (child)", "agent": "Agent", "physical_environment": "Physical environment",
        "social_environment": "Social environment", "narrative": "Narrative", "phase1_check": "Phase 1 check"}
TYPE = {"enum": "one value from the list", "multi_enum": "one or more values from the list",
        "text": f"short free text (under {s.FREE_TEXT_MAX} characters)"}

def codes(names) -> str:
    return ", ".join(f"`{n}`" for n in names)

def render() -> str:
    L = []
    w = L.append
    w(f"# Phase 2 codebook (schema {s.SCHEMA_VERSION})\n")
    w(f"Generated from `src/extract/schema.py`, the source of truth for the extraction prompt, the coding app and the QA checks. "
      f"Schema {s.SCHEMA_VERSION}, dated {s.SCHEMA_HISTORY[-1][1]}.\n")
    w("**Scope.** Unintentional (accidental) injury to children under 5, from the originating post only "
      "(comment threads were not collected).\n")
    w("**Framework.** Fields follow the Haddon matrix: three phases (pre-event, event, post-event) crossed with four factors "
      "(host, agent, physical environment, social environment). Interpretive fields and the Phase 1 audit sit outside the matrix. "
      "Controlled vocabularies are *adapted from* ICECI and extended with child-specific values; they are not transcribed from the "
      "canonical ICECI data dictionary, so check them against ICECI and the ICD-11 external-cause extension codes before claiming ICECI coding.\n")

    w("## How to code each field\n")
    w("Every field gets a status, a value and, unless not stated, a quote.\n")
    w("| Status | Meaning |\n|---|---|")
    for st in s.STATUSES:
        w(f"| `{st}` | {STATUS_HELP[st]} |")
    w("")
    w("**Rules**\n")
    w(f"- Code `{s.AUDIT_FIELD}` first. When it is {codes(sorted(s.HUMAN_SKIP_VALUES))}, every other field is left `not_stated` "
      "in human coding (the coding app locks them). The model still fills every field.")
    w("- Out of scope, so `not_an_injury_event`: intentional harm (assault, maltreatment, self-harm; harm from another child under 5 counts as unintentional), a child clearly aged 5 or older, "
      "and illness with no external cause. An unstated age is not grounds for exclusion.")
    w(f"- `{s.AUDIT_FIELD}` is never `not_stated`; a post you can't judge is `unclear`.")
    w(f"- {codes(s.CONDITIONAL_FALL_FIELDS)} are coded only when `mechanism_iceci` is a fall "
      f"({codes(sorted(s.FALL_MECHANISMS))}); otherwise they are `not_stated`.")
    w(f"- {codes(sorted(s.MULTI_VALUED))} can take several values.")
    w(f"- Free-text values stay under {s.FREE_TEXT_MAX} characters. Quotes are copied word for word; "
      "joined fragments are allowed, one continuous passage is preferred.")
    w(f"- Expected to be mostly `not_stated` (rare by design): {codes(sorted(s.RARE_BY_DESIGN_FIELDS))}.")
    w("- Near misses (`near_miss`) are included in the analysis as their own category, not pooled with injury events "
      "(study decisions, 2026-11-04 and 2026-11-11).")
    w(f"- Often `not_stated` because only the original post is coded: {codes(sorted(s.SCOPE_LIMITED_FIELDS))}.\n")

    w("## Fields at a glance\n")
    w("| Phase | Factor | Field | Type |\n|---|---|---|---|")
    for p, c, f in s.FIELDS:
        w(f"| {HEAD[p].split(' (')[0]} | {CELL.get(c, c)} | [`{f}`](#{f}) | {s.field_type(f).replace('_', ' ')} |")
    w("")

    phase = None
    for p, c, f in s.FIELDS:
        if p != phase:
            phase = p
            w(f"## {HEAD[p]}\n")
        w(f"### {f}\n")
        w(f"*{CELL.get(c, c)} · {TYPE[s.field_type(f)]}*\n")
        w(s.DEFINITIONS[f] + "\n")
        vals = s.VOCAB.get(f, ())
        if vals:
            notes = s.VALUE_NOTES.get(f, {})
            if any(notes.get(v) for v in vals):
                w("| Value | Note |\n|---|---|")
                for v in vals:
                    w(f"| `{v}` | {notes.get(v, '')} |")
            else:
                w("Values: " + codes(vals) + ".")
            w("")

    w("## Version history\n")
    w("| Version | Date | Change |\n|---|---|---|")
    for v, d, note in reversed(s.SCHEMA_HISTORY):
        w(f"| {v} | {d} | {note.replace('|', '/')} |")
    w("")
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="Output path; default: reports/codebook_<version>.md")
    args = ap.parse_args()
    out = Path(args.out) if args.out else find_repo_root(Path(__file__).resolve()) / "reports" / f"codebook_{s.SCHEMA_VERSION}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(), encoding="utf-8")
    print(f"Wrote codebook (schema {s.SCHEMA_VERSION}, {len(s.FIELDS)} fields) -> {out}")

if __name__ == "__main__":
    main()
