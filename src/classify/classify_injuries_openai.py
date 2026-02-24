# classify_injuries_openai.py
#!/usr/bin/env python3
"""
One-click VS Code classifier for child-injury Reddit posts (gpt-5-nano, Chat Completions JSON mode)
with client-side guardrails to enforce enums and consistency.
Now includes wall-clock timing (total, posts/sec, sec/post) and optional temperature control.
"""

import os, json, csv, gzip, re, time
from pathlib import Path
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from openai import OpenAI, BadRequestError

# ---------------- Config ----------------

MODEL_NAME = "gpt-5-nano"   # change here to test other models (e.g., "gpt-4.1-mini")
TEMPERATURE = None          # set e.g. 0.2 for models that support it; keep None for nano

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

# ---------------- Guardrails (enums + normalizer) ----------------

ENUMS = {
    "mechanism_of_injury": {
        "road_transport","fall","drowning","burn","scald","poisoning",
        "choking_or_suffocation","foreign_body_ingestion","cut_pierce",
        "struck_by_object","animal_related","other","unknown","not_applicable"
    },
    "nature_of_injury": {
        "fracture","laceration","contusion","burn","poisoning","asphyxiation",
        "internal_injury","dental_injury","multiple","other","unknown","not_applicable"
    },
    "body_region": {
        "head_face","neck","torso","arm_hand","leg_foot","multiple","unknown","not_applicable"
    },
    "age_group": {
        "newborn","infant","toddler","preschool","child_unspecified","unknown"
    }
}

def _norm_str(x):
    return (x or "").strip().lower()

def _norm_bool(x):
    s = _norm_str(str(x))
    return s in {"true","1","yes","y"}

def normalize_label(label: dict) -> dict:
    label["is_injury_event"] = _norm_bool(label.get("is_injury_event"))
    label["er_or_hospital_mentioned"] = _norm_bool(label.get("er_or_hospital_mentioned"))

    for k in ("mechanism_of_injury","nature_of_injury","body_region","age_group"):
        v = _norm_str(label.get(k))
        label[k] = v if v in ENUMS[k] else "unknown"

    label["rationale_short"] = (label.get("rationale_short") or "")[:280]

    if not label["is_injury_event"]:
        label["mechanism_of_injury"] = "not_applicable"
        label["nature_of_injury"]    = "not_applicable"
        label["body_region"]         = "not_applicable"
        label["er_or_hospital_mentioned"] = False
    else:
        for k in ("mechanism_of_injury","nature_of_injury","body_region"):
            if label[k] == "not_applicable":
                label[k] = "unknown"

    return label

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
  "mechanism_of_injury": one of ["road_transport","fall","drowning","burn","scald","poisoning","choking_or_suffocation","foreign_body_ingestion","cut_pierce","struck_by_object","animal_related","other","unknown","not_applicable"],
  "nature_of_injury": one of ["fracture","laceration","contusion","burn","poisoning","asphyxiation","internal_injury","dental_injury","multiple","other","unknown","not_applicable"],
  "body_region": one of ["head_face","neck","torso","arm_hand","leg_foot","multiple","unknown","not_applicable"],
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

    # Build kwargs with optional temperature; retry without if unsupported
    kwargs = dict(model=MODEL_NAME, messages=messages)
    if TEMPERATURE is not None:
        kwargs["temperature"] = float(TEMPERATURE)

    try:
        kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
    except (TypeError, BadRequestError) as e:
        emsg = str(e).lower()
        if "temperature" in emsg and "unsupported" in emsg:
            kwargs.pop("temperature", None)
        kwargs.pop("response_format", None)
        resp = client.chat.completions.create(**kwargs)

    text = resp.choices[0].message.content
    try:
        label = json.loads(_extract_json(text))
    except Exception as e:
        label = {
            "is_injury_event": False,
            "mechanism_of_injury": "unknown",
            "nature_of_injury": "unknown",
            "body_region": "unknown",
            "er_or_hospital_mentioned": False,
            "age_group": "unknown",
            "rationale_short": f"parser_error: {e.__class__.__name__}"
        }

    return normalize_label(label)

# ---------------- Main ----------------

def main():
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path)
    load_env(repo_root)
    require_api_key()

    raw_dir = repo_root / "data" / "raw"
    inp_path = newest_raw_file(raw_dir)
    print(f"Using input: {inp_path}")
    print(f"Model: {MODEL_NAME} | temperature: {TEMPERATURE if TEMPERATURE is not None else 'default'}")

    out_dir = repo_root / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = stem_for_output(inp_path)
    out_csv = out_dir / f"{base}_labels.csv"
    out_jsonl = out_dir / f"{base}_labels.jsonl"
    out_metrics = out_dir / f"{base}_timing.json"  # timing summary

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
        "is_injury_event","mechanism_of_injury","nature_of_injury","body_region",
        "er_or_hospital_mentioned","age_group","rationale_short"
    ]

    ALLOWED_LABEL_KEYS = {
        "is_injury_event","mechanism_of_injury","nature_of_injury","body_region",
        "er_or_hospital_mentioned","age_group","rationale_short"
    }

    def filter_label_keys(label: dict) -> dict:
        out = {k: label.get(k, "") for k in ALLOWED_LABEL_KEYS}
        if out["is_injury_event"] in ("", None): out["is_injury_event"] = False
        if out["er_or_hospital_mentioned"] in ("", None): out["er_or_hospital_mentioned"] = False
        out["rationale_short"] = (out.get("rationale_short") or "")[:280]
        return out

    kept = rejected = 0
    processed = 0

    # ---- start timing just before classification loop
    t0 = time.perf_counter()

    with open(out_csv, "w", newline="", encoding="utf-8") as csv_fp, \
         open(out_jsonl, "w", encoding="utf-8") as jsonl_fp:

        writer = csv.DictWriter(csv_fp, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()

        for idx, rec in enumerate(posts, 1):
            title = rec.get("title") or ""
            body  = rec.get("selftext") or ""

            label = classify_post(client, title, body)
            clean_label = filter_label_keys(label)

            extra = sorted(set(label.keys()) - ALLOWED_LABEL_KEYS)
            if extra:
                print(f"[warn] extra label keys ignored at idx {idx}: {extra}")

            writer.writerow({
                "id": rec.get("id"),
                "subreddit": rec.get("subreddit"),
                "created_utc": rec.get("created_utc"),
                "permalink": rec.get("permalink"),
                "title": title,
                **clean_label
            })

            jsonl_fp.write(json.dumps({**rec, "**labels": clean_label}, ensure_ascii=False) + "\n")

            processed += 1
            if clean_label.get("is_injury_event"):
                kept += 1
            else:
                rejected += 1

            if idx % 25 == 0 or idx == total:
                print(f"[{idx}/{total}] kept={kept} rejected={rejected}")

    # ---- stop timing after loop
    t1 = time.perf_counter()
    elapsed_sec = t1 - t0
    sec_per_post = elapsed_sec / processed if processed else float("nan")
    posts_per_sec = processed / elapsed_sec if elapsed_sec > 0 else float("nan")

    print("\nTiming summary")
    print(f"  processed posts : {processed}")
    print(f"  total time      : {elapsed_sec:.2f} s")
    print(f"  posts/sec       : {posts_per_sec:.3f}")
    print(f"  sec/post        : {sec_per_post:.3f}")

    try:
        with open(out_metrics, "w", encoding="utf-8") as mfp:
            json.dump({
                "input_file": str(inp_path),
                "output_csv": str(out_csv),
                "output_jsonl": str(out_jsonl),
                "processed": processed,
                "kept": kept,
                "rejected": rejected,
                "elapsed_seconds": elapsed_sec,
                "posts_per_second": posts_per_sec,
                "seconds_per_post": sec_per_post,
                "model": MODEL_NAME,
                "temperature": TEMPERATURE if TEMPERATURE is not None else "default",
            }, mfp, ensure_ascii=False, indent=2)
        print(f"\nSaved timing metrics -> {out_metrics}")
    except Exception as e:
        print(f"[warn] could not write timing metrics: {e}")

    print(f"\nDone.\nCSV: {out_csv}\nJSONL: {out_jsonl}")

if __name__ == "__main__":
    main()
