import praw
import os
import itertools
import datetime
import time
import json, gzip, re, hashlib, tempfile, shutil

from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)


SUBS = "Parenting+Mommit+Daddit+NewParents+ChildSafety+BabyBumps+Nanny+Daycare+AskParents+BeyondTheBump"


EVENTS = [
    "fell", "fall", "slipped", "hit head", "burn", "scald", "choked", "swallowed",
    "poison", "detergent pod", "button battery", "magnet", "stitches", "fracture",
    "cut", "concussion", "bumped head", "ER", "A&E", "ED"
]
AGES = ["baby", "infant", "newborn", "toddler", "1yo", "2 yo", "3 yo", "18 months", "LO", "DD", "DS"]

EXCLUDES = ["lawyer", "insurance", "workout", "sports", "MLB", "NFL"]

def build_query(event, age):
    # Use title: for high-precision variants when useful
    core = f'("{event}" AND {age})'
    filters = 'self:yes nsfw:no'
    negatives = " ".join(f'-{w}' for w in EXCLUDES)
    return f'{core} {filters} {negatives}'

def search_terms(subs, events, ages, limit_per_query=100, cutoff_date="2020-01-01"):
    cutoff_ts = datetime.datetime.fromisoformat(cutoff_date).timestamp()
    seen_ids = set()
    sub = reddit.subreddit(subs)

    for event, age in itertools.product(events, ages):
        q = build_query(event, age)
        # Sort by new to more quickly bump into older posts; adjust as needed
        for post in sub.search(q, sort="new", time_filter="all", limit=limit_per_query):
            if post.id in seen_ids:
                continue
            if post.created_utc >= cutoff_ts:
                seen_ids.add(post.id)
                yield {
                    "id": post.id,
                    "title": post.title,
                    "selftext": post.selftext or "",
                    "created_utc": post.created_utc,
                    "permalink": post.permalink,
                    "subreddit": str(post.subreddit),
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "matched_event": event,
                    "matched_age": age
                }
        # Be nice to the API if you’re doing lots of queries
        time.sleep(1.1)

# ------------ minimal privacy scrub (best-effort, not perfect) ------------
_EMAIL_RE = re.compile(r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b', re.I)
_PHONE_RE = re.compile(r'(?<!\d)(?:\+?\d[\s-]?){6,15}\d(?!\d)')   # broad intl matcher
_URL_RE   = re.compile(r'(https?://\S+|www\.\S+)', re.I)
_HANDLE_RE= re.compile(r'(?<!\w)@[A-Za-z0-9_]{2,}', re.I)

def scrub_text(t: str) -> str:
    if not t:
        return ""
    t = _EMAIL_RE.sub("[redacted-email]", t)
    t = _PHONE_RE.sub("[redacted-phone]", t)
    t = _URL_RE.sub("[redacted-url]", t)
    t = _HANDLE_RE.sub("[redacted-handle]", t)
    return t

def sanitise_row(row: dict) -> dict:
    """Keep only approved fields, scrub text, and add bookkeeping."""
    kept = {
        "id": row.get("id"),
        "subreddit": row.get("subreddit"),
        "title": scrub_text(row.get("title") or ""),
        "selftext": scrub_text(row.get("selftext") or ""),
        "created_utc": row.get("created_utc"),
        "permalink": row.get("permalink"),
        "score": row.get("score"),
        "num_comments": row.get("num_comments"),
        "matched_event": row.get("matched_event"),
        "matched_age": row.get("matched_age"),
        "schema_version": 1,
    }
    body = (kept["title"] + "\n" + kept["selftext"]).strip()
    kept["content_hash"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return kept


# ------------ JSONL.gz appender with atomic write ------------
def append_jsonl_gz(record: dict, out_path: str):
    """
    Append one JSON object per line (newline-delimited) to a gzipped file.
    Uses a small temp file and os.replace to avoid partial writes on crashes.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"

    # Write to a tiny temp gz and concatenate safely to the target
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(out_path.parent)) as tmp:
        tmp_name = tmp.name
        with gzip.GzipFile(fileobj=tmp, mode="wb", mtime=0) as gz:
            gz.write(line.encode("utf-8"))

    # If the target doesn't exist yet, just move temp into place.
    # If it exists, append the bytes (binary concat of gzip members is valid).
    if not out_path.exists():
        shutil.move(tmp_name, out_path)
    else:
        with open(out_path, "ab") as fout, open(tmp_name, "rb") as fin:
            shutil.copyfileobj(fin, fout)
        Path(tmp_name).unlink(missing_ok=True)

# ------------ choose an output file name ------------
RUN_STAMP = datetime.datetime.utcnow().strftime("%Y%m%d")
OUTFILE = f"data/raw/reddit_child_injury_{RUN_STAMP}.jsonl.gz"

# Run it (stream -> print -> append)
saved = 0
for i, row in enumerate(search_terms(SUBS, EVENTS, AGES, limit_per_query=100, cutoff_date="2020-01-01"), 1):
    safe = sanitise_row(row)
    print(i, safe["subreddit"], safe["title"], safe["selftext"][:120].replace("\n"," ") + ("..." if len(safe["selftext"]) > 120 else ""))
    append_jsonl_gz(safe, OUTFILE)
    saved += 1

print(f"\nSaved {saved} records to {OUTFILE}")
