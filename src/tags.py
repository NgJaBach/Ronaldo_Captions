"""The tagger. Gives every caption situation tags (setback, win, family, ...) so you can filter by what a post is about.

  python src/tags.py             # (re)builds the `tags` table in data/ronaldo.db
  python src/tags.py --report    # tag counts

How: zero-shot, CPU-only. Each tag is a handful of example phrases written in Ronaldo's own voice (taken from real
captions). A caption's score for a tag is its best cosine similarity to any phrase (same embedding model as search).
A caption gets the tag when the score passes that tag's threshold in TAGS. `sponsor` is rule-based (links, @brands,
"download", "available") because ads are better caught by their wording than by their meaning.
Each tag needs a z-score of 2.0 or more across captions (see docs/context.md for how thresholds were chosen and checked),
so treat tags as hints, not ground truth.
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "ronaldo.db"
NPY = ROOT / "data" / "embeddings.npy"
META = ROOT / "data" / "embeddings.json"

# tag -> {"min": z-score threshold, "phrases": [...]}. 2.0 = two standard deviations above the typical caption.
TAGS = {
    "setback": {"min": 2.0, "phrases": [
        "Not the result we wanted, but we will come back stronger.",
        "Disappointed with the result, but we stay focused on the next game.",
        "A tough night. We keep fighting and we will rise again.",
        "Every setback is a lesson for a stronger comeback.",
        "Sometimes the dream has to wait. Proud of this team.",
        "It wasn't our day. Head up and keep working.",
    ]},
    "win": {"min": 1.8, "phrases": [
        "Great win! Three important points for the team.",
        "Victory tonight! Thank you to the fans for the support.",
        "We did it! What a game, what a result.",
        "Important win, well done lads.",
    ]},
    "trophy": {"min": 2.0, "phrases": [
        "Champions! Another trophy for the team.",
        "We won the title! Proud of everyone.",
        "One more trophy. The hard work pays off.",
        "Winners of the cup, a dream come true.",
    ]},
    "milestone": {"min": 2.0, "phrases": [
        "A new record. History made.",
        "Proud of this milestone in my career, 900 goals.",
        "Another historic achievement, grateful for the journey.",
        "Records keep falling, and I keep going.",
    ]},
    "gratitude": {"min": 2.0, "phrases": [
        "Thank you all for your incredible support.",
        "Grateful for everyone who believed in me.",
        "Thank you to my teammates, the staff and the fans.",
        "Blessed and thankful for this moment.",
    ]},
    "family": {"min": 2.0, "phrases": [
        "Family is everything. Quality time with my loves.",
        "Happy birthday to my beautiful son. Papa loves you.",
        "Happy Mother's Day to the special women in my life.",
        "Proud of my children, my family and Georgina.",
    ]},
    "training": {"min": 2.0, "phrases": [
        "Back to work. Hard work never stops.",
        "Training session done. Recovery time and gym.",
        "No shortcuts. Every workout counts, every day.",
        "Preparing for the next game, working hard.",
    ]},
    "mindset": {"min": 2.0, "phrases": [
        "Never give up. Believe in yourself and keep dreaming.",
        "Hard work, discipline and determination beat talent.",
        "The only way is forward. Keep pushing your limits.",
        "Success is a mindset: stay hungry and humble.",
    ]},
    "team_spirit": {"min": 2.0, "phrases": [
        "Together we are stronger. Great team spirit.",
        "All together until the end, one team, one family.",
        "Proud of the squad's spirit and unity.",
    ]},
    "fans": {"min": 2.0, "phrases": [
        "What an incredible reception from the fans.",
        "The best fans in the world, thank you for the atmosphere.",
        "Amazing support from the stands tonight.",
    ]},
    "tribute": {"min": 2.5, "phrases": [
        "Rest in peace. My deepest condolences to the family.",
        "Heartbreaking news. Our thoughts are with everyone affected.",
        "Sending love and strength to those who are suffering.",
    ]},
    "greeting": {"min": 2.0, "phrases": [
        "Merry Christmas to you and your families!",
        "Happy New Year! Wishing you all health and happiness.",
        "Eid Mubarak to everyone celebrating.",
        "Good morning everyone, have a great day.",
    ]},
    "injury": {"min": 2.5, "phrases": [
        "Recovering well after the injury. Back soon.",
        "A small injury, but I will be back stronger.",
    ]},
}

SPONSOR_RX = re.compile(
    r"(https?://|www\.|\.com\b|#ad\b|#sponsored|#partner|download|link in bio|available (now|on|at)|"
    r"pre-?order|shop\b|collection|discount|promo|code\b|giveaway|@\w+.*\b(app|game|bag|watch|fragrance|brand)\b)",
    re.I)


def score_all():
    """Return (codes, {tag: z-scores per caption}).

    The embedding model's cosines are compressed (almost every caption scores 0.8+ against everything), so raw cosines
    are useless as thresholds. Each tag's best-phrase cosine is converted to a z-score across the captions that have
    real words; a tag then means "unusually close to this tag's phrases compared with the other captions".
    Captions without words (emoji only) score the minimum everywhere and are never tagged.
    """
    from sentence_transformers import SentenceTransformer
    import transformers
    transformers.logging.set_verbosity_error()
    meta = json.loads(META.read_text(encoding="utf-8"))
    codes = [r["code"] for r in meta["rows"]]
    vecs = np.load(NPY).astype(np.float32)
    con = sqlite3.connect(DB)
    caps = dict(con.execute("SELECT code, caption FROM posts"))
    texty = np.array([sum(ch.isalpha() for ch in caps.get(c, "")) >= 3 for c in codes])
    model = SentenceTransformer(meta["model"], device="cpu")
    out = {}
    for tag, cfg in TAGS.items():
        p = model.encode(["query: " + s for s in cfg["phrases"]], normalize_embeddings=True)
        raw = (vecs @ p.T).max(axis=1)
        z = (raw - raw[texty].mean()) / raw[texty].std()
        z[~texty] = -9.0
        out[tag] = z
    return codes, out


def build() -> None:
    codes, scores = score_all()
    con = sqlite3.connect(DB)
    con.executescript("DROP TABLE IF EXISTS tags; CREATE TABLE tags (code TEXT, tag TEXT, score REAL, PRIMARY KEY (code, tag));")
    caps = dict(con.execute("SELECT code, caption FROM posts"))
    rows = []
    for tag, cfg in TAGS.items():
        for code, s in zip(codes, scores[tag]):
            if s >= cfg["min"]:
                rows.append((code, tag, float(s)))
    for code in codes:
        if SPONSOR_RX.search(caps.get(code, "")):
            rows.append((code, "sponsor", 1.0))
    con.executemany("INSERT OR REPLACE INTO tags VALUES (?,?,?)", rows)
    con.commit()
    report()


def report() -> None:
    con = sqlite3.connect(DB)
    n = con.execute("SELECT COUNT(*) FROM posts WHERE caption != ''").fetchone()[0]
    print(f"{n} captions; tag counts:")
    for tag, c in con.execute("SELECT tag, COUNT(*) FROM tags GROUP BY tag ORDER BY 2 DESC"):
        print(f"  {tag:12} {c}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    report() if "--report" in sys.argv else build()
