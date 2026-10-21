#!/usr/bin/env python3
"""
Shared I/O helpers for the Phase 2 scripts. The first block is copied verbatim
from src/classify/classify_injuries_openai.py so the two phases read identically.
"""

import gzip, json, re
from pathlib import Path
from typing import Any, Dict, List

# ---------------- Phase 1 helpers (verbatim) ----------------

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

# ---------------- Phase 2 input helpers ----------------

def newest_matching(dir_: Path, pattern: str) -> Path:
    candidates = list(dir_.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern} in {dir_}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open_maybe_gzip(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out

def get_labels(rec: Dict[str, Any]) -> Dict[str, Any]:
    # Phase 1 writes {**rec, "**labels": clean_label} — the asterisks are part of the key.
    for key in ("**labels", "labels"):
        if isinstance(rec.get(key), dict):
            return rec[key]
    return rec

def is_truthy(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes"}

def is_confirmed(rec: Dict[str, Any]) -> bool:
    return is_truthy(get_labels(rec).get("is_injury_event"))

def require_selftext(records: List[Dict[str, Any]], path: Path):
    # The Phase 1 labels CSV has no selftext. Pointing Phase 2 at it would not error —
    # it would silently return not_stated for nearly every field. Fail loudly instead.
    if not records and path.suffix == ".csv":
        raise SystemExit(f"{path} is a CSV. Phase 2 needs the Phase 1 *_labels.jsonl, which carries selftext.")
    if not records:
        raise SystemExit(f"No records read from {path}")
    if "selftext" not in records[0]:
        raise SystemExit(
            f"First record in {path} has no 'selftext' field.\n"
            "Phase 2 needs the post body. Use the Phase 1 *_labels.jsonl (or a file derived "
            "from it), never the *_labels.csv."
        )
