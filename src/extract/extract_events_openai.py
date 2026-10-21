#!/usr/bin/env python3
"""
Phase 2 extractor: Haddon/ICECI-adapted event chain for confirmed child-injury posts
(Chat Completions JSON mode), with client-side guardrails from schema.py.

Input must be a JSONL derived from Phase 1's *_labels.jsonl (it carries selftext);
the Phase 1 CSV has no post body and is rejected. Phase 1's rationale_short is
deliberately NOT sent to the model, to avoid anchoring extraction on Phase 1's answer.

The prompt does not tell the model that Phase 1 accepted the post, so the
injury_event_check audit field is an independent check on Phase 1 false positives.
"""

import os, json, csv, time, argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from openai import OpenAI, BadRequestError

from common import (find_repo_root, load_env, load_jsonl, newest_matching, stem_for_output,
                    _extract_json, get_labels, is_confirmed, require_selftext)
from schema import (SCHEMA_VERSION, FIELD_NAMES, MULTI_VALUED, blank_cell,
                    normalize_extraction, render_field_spec)

# ---------------- Config ----------------

DEFAULT_MODEL = "gpt-5-nano"   # placeholder for consistency with Phase 1; the pilot decides
BODY_MAX = 6000                # longer than Phase 1's 4000: circumstantial detail lives in the body

SYSTEM_PROMPT = """You are a careful public-health coder extracting structured detail from Reddit posts.
1. Posts come from a corpus of parenting posts screened for child injury events. Extract the event chain as the post describes it. Separately, record in injury_event_check your own independent judgement of whether the post describes a real child injury event; that judgement does not change how you fill the other fields.
2. Every field is an object {"value": ..., "evidence_span": ..., "status": ...} where status is one of:
   - "stated": the post says it explicitly. evidence_span is REQUIRED.
   - "inferred": the post implies it. evidence_span must be the text supporting the inference.
   - "not_stated": the post gives no support. value and evidence_span must both be null.
3. NEVER invent an evidence span. It must be ONE contiguous passage copied character-for-character from the title or body. Do not join separate passages with "..." or line breaks; pick the single most informative passage.
4. Prefer "not_stated" over a guess. Do not assume supervision status, place, or caregiver activity from typical parenting context — only from what THIS post says.
5. Use only the allowed values for enumerated fields. Use "unspecified" or "other_specified" when nothing fits.
6. Free-text values are short paraphrases in plain English (not snake_case). Do not copy long passages into value.
7. fall_height and landing_surface apply only when mechanism_iceci is a fall mechanism; otherwise mark them not_stated.
"""

# ---------------- Helpers ----------------

def require_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Add to .env at repo root.")

def parse_temperature(s: Optional[str]) -> Optional[float]:
    if s is None or str(s).strip().lower() in {"none", "default", ""}:
        return None
    return float(s)

def build_user_prompt(title: str, body: str) -> str:
    body = (body or "").strip()
    if len(body) > BODY_MAX:
        body = body[:BODY_MAX] + " ...[truncated]"
    return f"""Extract the event chain from this Reddit post.

TITLE:
{title or ''}

BODY:
{body}

Return a single flat JSON object with exactly these {len(FIELD_NAMES)} keys (bare field names such as
"activity_when_injured" — section headings are not keys), each an object
{{"value": ..., "evidence_span": ..., "status": "stated"|"inferred"|"not_stated"}}:

{render_field_spec()}

No prose. No code fences. JSON only.
"""

def cell_to_csv(name: str, cell: Dict[str, Any]) -> str:
    v = cell.get("value")
    if v is None:
        return ""
    if name in MULTI_VALUED and isinstance(v, list):
        return "|".join(v)
    return str(v)

# ---------------- Model call (Chat Completions JSON mode) ----------------

def usage_of(resp) -> Dict[str, int]:
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    details = getattr(u, "completion_tokens_details", None)
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "reasoning_tokens": (getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
    }

@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(initial=1, max=12))
def call_model(client: OpenAI, model: str, temperature: Optional[float], reasoning_effort: Optional[str],
               title: str, body: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_prompt(title, body)},
    ]

    # Build kwargs with optional temperature / reasoning effort; retry without if unsupported
    kwargs = dict(model=model, messages=messages)
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    try:
        kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
    except (TypeError, BadRequestError) as e:
        emsg = str(e).lower()
        if "temperature" in emsg and "unsupported" in emsg:
            kwargs.pop("temperature", None)
        if "reasoning_effort" in emsg:
            kwargs.pop("reasoning_effort", None)
        kwargs.pop("response_format", None)
        resp = client.chat.completions.create(**kwargs)

    return resp.choices[0].message.content, getattr(resp, "model", None), usage_of(resp)

def extract_event(client: OpenAI, model: str, temperature: Optional[float], reasoning_effort: Optional[str],
                  title: str, body: str):
    text, resolved_model, usage = call_model(client, model, temperature, reasoning_effort, title, body)
    try:
        raw = json.loads(_extract_json(text or ""))
    except Exception as e:
        result = {
            "fields": {name: blank_cell() for name in FIELD_NAMES},
            "violations": [{"field": None, "code": "parser_error", "detail": e.__class__.__name__}],
        }
        return result, False, resolved_model, usage, text
    return normalize_extraction(raw, f"{title}\n{body}"), True, resolved_model, usage, text

# ---------------- Main ----------------

def main():
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path)
    load_env(repo_root)

    ap = argparse.ArgumentParser(description="Phase 2 Haddon/ICECI event extraction.")
    ap.add_argument("--in", dest="inp", default=None, help="Input JSONL; default: newest data/interim/confirmed_events_n*.jsonl")
    ap.add_argument("--outdir", default=None, help="Output directory; default: data/interim/")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", default=None, help="e.g. 0.2; 'None' for models that reject it")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N confirmed events")
    ap.add_argument("--tag", default=None, help="Suffix for output filenames (e.g. nano, mini)")
    ap.add_argument("--reasoning-effort", default=None, help="GPT-5-family only, e.g. minimal|low|medium|high; default: model default")
    ap.add_argument("--workers", type=int, default=1, help="Concurrent API calls; output order is preserved")
    args = ap.parse_args()

    require_api_key()
    temperature = parse_temperature(args.temperature)

    interim = repo_root / "data" / "interim"
    inp_path = Path(args.inp).expanduser().resolve() if args.inp else newest_matching(interim, "confirmed_events_n*.jsonl")
    print(f"Using input: {inp_path}")
    print(f"Model: {args.model} | temperature: {temperature if temperature is not None else 'default'} | "
          f"reasoning_effort: {args.reasoning_effort or 'default'} | workers: {args.workers} | schema: {SCHEMA_VERSION}")

    out_dir = Path(args.outdir).expanduser().resolve() if args.outdir else interim
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{stem_for_output(inp_path)}_extractions" + (f"_{args.tag}" if args.tag else "")
    out_csv = out_dir / f"{base}.csv"
    out_jsonl = out_dir / f"{base}.jsonl"
    out_metrics = out_dir / f"{base}_timing.json"

    records = load_jsonl(inp_path)
    require_selftext(records, inp_path)
    events = [r for r in records if is_confirmed(r)]
    print(f"Loaded {len(records)} records -> {len(events)} confirmed injury events")
    if args.limit is not None:
        events = events[:args.limit]
        print(f"--limit {args.limit}: processing {len(events)}")

    client = OpenAI()  # reads OPENAI_API_KEY

    csv_fields = ["id", "subreddit", "created_utc", "permalink", "title",
                  "extraction_schema_version", "model", "parse_ok"]
    for name in FIELD_NAMES:
        csv_fields += [name, f"{name}__status", f"{name}__span"]
    # Phase 1 model's own note — reference only for human coders, never sent to the Phase 2 model
    csv_fields += ["n_violations", "violation_codes", "phase1_mechanism_of_injury", "phase1_model_rationale"]

    tally = Counter()
    resolved_models = Counter()
    tokens = Counter()
    processed = parse_errors = with_violations = 0
    total = len(events)

    def run_one(rec):
        return extract_event(client, args.model, temperature, args.reasoning_effort,
                             rec.get("title") or "", rec.get("selftext") or "")

    # ---- start timing just before extraction loop
    t0 = time.perf_counter()

    with open(out_csv, "w", newline="", encoding="utf-8") as csv_fp, \
         open(out_jsonl, "w", encoding="utf-8") as jsonl_fp, \
         ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:

        writer = csv.DictWriter(csv_fp, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()

        # pool.map yields in input order, so outputs match input order at any --workers
        for idx, (rec, out) in enumerate(zip(events, pool.map(run_one, events)), 1):
            title = rec.get("title") or ""
            body  = rec.get("selftext") or ""
            p1 = get_labels(rec)

            result, parse_ok, resolved_model, usage, raw_text = out
            fields, violations = result["fields"], result["violations"]

            tokens.update(usage)
            resolved_models[resolved_model or args.model] += 1
            codes = [v["code"] for v in violations]
            tally.update(codes)
            processed += 1
            parse_errors += (not parse_ok)
            with_violations += bool(violations)

            row = {
                "id": rec.get("id"),
                "subreddit": rec.get("subreddit"),
                "created_utc": rec.get("created_utc"),
                "permalink": rec.get("permalink"),
                "title": title,
                "extraction_schema_version": SCHEMA_VERSION,
                "model": resolved_model or args.model,
                "parse_ok": parse_ok,
                "n_violations": len(violations),
                "violation_codes": "|".join(sorted(set(codes))),
                "phase1_mechanism_of_injury": p1.get("mechanism_of_injury"),
                "phase1_model_rationale": p1.get("rationale_short"),
            }
            for name in FIELD_NAMES:
                cell = fields[name]
                row[name] = cell_to_csv(name, cell)
                row[f"{name}__status"] = cell["status"]
                row[f"{name}__span"] = cell["evidence_span"] or ""
            writer.writerow(row)

            jsonl_fp.write(json.dumps({
                "id": rec.get("id"),
                "subreddit": rec.get("subreddit"),
                "created_utc": rec.get("created_utc"),
                "permalink": rec.get("permalink"),
                "title": title,
                "selftext": body,
                "extraction_schema_version": SCHEMA_VERSION,
                "model": resolved_model or args.model,
                "parse_ok": parse_ok,
                "usage": usage,
                "fields": fields,
                "violations": violations,
                # raw model output, so guardrails can be re-applied after a schema change without re-calling the API
                "raw_response": raw_text,
                "phase1_labels": p1 if p1 is not rec else {},
            }, ensure_ascii=False) + "\n")
            csv_fp.flush(); jsonl_fp.flush()

            if idx % 25 == 0 or idx == total:
                print(f"[{idx}/{total}] parse_errors={parse_errors} events_with_violations={with_violations}")

    # ---- stop timing after loop
    t1 = time.perf_counter()
    elapsed_sec = t1 - t0
    sec_per_event = elapsed_sec / processed if processed else float("nan")

    print("\nTiming summary")
    print(f"  processed events : {processed}")
    print(f"  total time       : {elapsed_sec:.2f} s")
    print(f"  sec/event        : {sec_per_event:.3f}")
    if processed and tokens:
        print(f"  tokens/event     : in {tokens['prompt_tokens'] / processed:.0f} | "
              f"out {tokens['completion_tokens'] / processed:.0f} (reasoning {tokens['reasoning_tokens'] / processed:.0f})")

    print("\nViolation tally by code")
    if tally:
        for code, k in tally.most_common():
            print(f"  {code:<28}{k:>6}")
    else:
        print("  (none)")

    try:
        with open(out_metrics, "w", encoding="utf-8") as mfp:
            json.dump({
                "input_file": str(inp_path),
                "output_csv": str(out_csv),
                "output_jsonl": str(out_jsonl),
                "processed": processed,
                "parse_errors": parse_errors,
                "events_with_violations": with_violations,
                "elapsed_seconds": elapsed_sec,
                "seconds_per_event": sec_per_event,
                "model": args.model,
                "model_resolved": dict(resolved_models),
                "temperature": temperature if temperature is not None else "default",
                "reasoning_effort": args.reasoning_effort or "default",
                "workers": args.workers,
                "token_usage": dict(tokens),
                "extraction_schema_version": SCHEMA_VERSION,
                "body_max_chars": BODY_MAX,
                "limit": args.limit,
                "tag": args.tag,
                "violation_tally": dict(tally.most_common()),
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, mfp, ensure_ascii=False, indent=2)
        print(f"\nSaved timing metrics -> {out_metrics}")
    except Exception as e:
        print(f"[warn] could not write timing metrics: {e}")

    print(f"\nDone.\nCSV: {out_csv}\nJSONL: {out_jsonl}")

if __name__ == "__main__":
    main()
