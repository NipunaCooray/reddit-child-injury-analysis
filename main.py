import praw
import os
import itertools
import datetime
import time
from dotenv import load_dotenv


load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)


SUBS = "Parenting+Mommit"
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

# Run it
for i, row in enumerate(search_terms(SUBS, EVENTS, AGES, limit_per_query=10, cutoff_date="2025-01-01"), 1):
    print(i, row["subreddit"], row["title"], row["selftext"])
