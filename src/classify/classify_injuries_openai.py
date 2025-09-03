#!/usr/bin/env python3
"""
One-click VS Code classifier for child-injury Reddit posts (gpt-5-nano, Chat Completions JSON mode).

Layout:
reddit-injury/
  data/
    raw/
    interim/
    processed/
  src/
    search/
    classify/
      classify_injuries_openai.py
"""

import os, json, csv, gzip, re
from pathlib import Path
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from openai import OpenAI, BadRequestError

# ---------------- Config ----------------

MODEL_NAME = "gpt-5-nano"  # good price/perf for classification

SYSTEM_PROMPT = """You are a careful public-health coder.
Task: Decide if a Reddit post describes a REAL child injury incident and classify it.
Definitions:
- child = under 5 years unless clearly older; if unclear, use 'child_unspecified'.
- injury event = an accidental harmful incident (e.g., fall, burn, ingestion, bite, fracture, head injury, choking, poisoning) actually happening or clearly suspected.
Exclusions:
- sleep/nap issues; crafts; general parenting vents; relationship/feeding/pacifier topics;
- metaphor (e.g., “fell in love”, “fall behind”);
- purely hypothetical (“what if my child fell”), advice-only without an actual event, or adult injuries (unless they directly concern harm to the child).
Label conservatively: if uncertain the post is a real incident, set is_injury_event=false.
Avoid exact long quotes; keep reasoning generic.
"""

# ---------------- Minimal helpers ----------------

def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "data" / "raw").exists():
            return p
    return start.parents[2] if len(start.parents) >= 3 else start

def load_env(repo_root: Path):
    try:
        from dotenv import load_dotenv
        load_dotenv(repo_root / ".env")
    except Exception:
        pass

def require_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Add to .env at repo root.")

def newest_raw_file(raw_dir: Path) -> Path:
    candidates = list(raw_dir.glob("*.jsonl")) + list(raw_dir.glob("*.jsonl.gz"))
    if not candidates:
        raise FileNotFoundError(f"No .jsonl/.jsonl.gz files in {raw_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def open_maybe_gzip(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, "r", encoding="utf-8")

def stem_for_output(path: Path) -> str:
    name = path.name
    if name.endswith(".jsonl.gz"): return name[:-9]
    if name.endswith(".jsonl"):    return name[:-6]
    return path.stem

def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output")
    return m.group(0)

# ---------------- Model call (Chat Completions JSON mode) ----------------

@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(initial=1, max=12))
def classify_post(client: OpenAI, title: str, body: str) -> Dict[str, Any]:
    body = (body or "").strip()
    if len(body) > 4000:
        body = body[:4000] + " ...[truncated]"

    user_prompt = f"""Classify this Reddit post.

TITLE:
{title or ''}

BODY:
{body}

Return a single JSON object with exactly these keys and allowed values:
{{
  "is_injury_event": boolean,
  "primary_injury_type": one of ["fall","head_injury","burn","scald","choking","swallowing","poisoning","ingestion","cut_laceration","fracture","dental","animal_bite","drowning","other","unknown"],
  "body_region": one of ["head_face","neck","torso","arm_hand","leg_foot","multiple","unknown"],
  "severity": one of ["minor","moderate","severe","unknown"],
  "er_or_hospital_mentioned": boolean,
  "age_group": one of ["newborn","infant","toddler","preschool","child_unspecified","unknown"],
  "rationale_short": string (<=280 chars; paraphrase only)
}}
No prose. No code fences. JSON only.
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    # Do NOT send temperature (gpt-5-nano only supports default)
    kwargs = dict(model=MODEL_NAME, messages=messages)

    # Try JSON mode; if SDK/model rejects response_format, retry without it.
    try:
        kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
    except (TypeError, BadRequestError):
        kwargs.pop("response_format", None)
        resp = client.chat.completions.create(**kwargs)

    text = resp.choices[0].message.content
    try:
        label = json.loads(_extract_json(text))
    except Exception as e:
        label = {
            "is_injury_event": False,
            "primary_injury_type": "unknown",
            "body_region": "unknown",
            "severity": "unknown",
            "er_or_hospital_mentioned": False,
            "age_group": "unknown",
            "rationale_short": f"parser_error: {e.__class__.__name__}"
        }

    label["rationale_short"] = (label.get("rationale_short") or "")[:280]
    return label

# ---------------- Main ----------------

def main():
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path)
    load_env(repo_root)
    require_api_key()

    raw_dir = repo_root / "data" / "raw"
    inp_path = newest_raw_file(raw_dir)
    print(f"Using input: {inp_path}")

    out_dir = repo_root / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = stem_for_output(inp_path)
    out_csv = out_dir / f"{base}_labels.csv"
    out_jsonl = out_dir / f"{base}_labels.jsonl"

    client = OpenAI()  # reads OPENAI_API_KEY

    # Load posts
    posts = []
    with open_maybe_gzip(inp_path) as f:
        for line in f:
            try:
                posts.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total = len(posts)
    print(f"Loaded {total} posts")

    csv_fields = [
        "id","subreddit","created_utc","permalink","title",
        "is_injury_event","primary_injury_type","body_region","severity",
        "er_or_hospital_mentioned","age_group","rationale_short"
    ]

    kept = rejected = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as csv_fp, \
         open(out_jsonl, "w", encoding="utf-8") as jsonl_fp:

        writer = csv.DictWriter(csv_fp, fieldnames=csv_fields)
        writer.writeheader()

        for idx, rec in enumerate(posts, 1):
            title = rec.get("title") or ""
            body  = rec.get("selftext") or ""

            label = classify_post(client, title, body)

            writer.writerow({
                "id": rec.get("id"),
                "subreddit": rec.get("subreddit"),
                "created_utc": rec.get("created_utc"),
                "permalink": rec.get("permalink"),
                "title": title,
                **label
            })

            jsonl_fp.write(json.dumps({**rec, "**labels": label}, ensure_ascii=False) + "\n")

            if label.get("is_injury_event"):
                kept += 1
            else:
                rejected += 1

            if idx % 25 == 0 or idx == total:
                print(f"[{idx}/{total}] kept={kept} rejected={rejected}")

    print(f"Done.\nCSV: {out_csv}\nJSONL: {out_jsonl}")

if __name__ == "__main__":
    main()
