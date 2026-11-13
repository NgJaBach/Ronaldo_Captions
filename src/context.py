"""The context desk. Attaches to each post the match it most likely follows, so you can ask for "after a loss".

  python src/context.py            # rebuilds the `matches` and `post_match` tables in data/ronaldo.db (idempotent)
  python src/context.py --report   # a summary of what was joined

Rule: a post is "after" the most recent match whose date is the post's date or up to WINDOW days earlier (dates in UTC).
Honest limits: the match date is local, the post time is UTC, and kickoff times are not used, so a morning post on match
day can be mislabelled as "after"; a post within the window may also be about something else entirely (an ad, a family
photo). Treat the label as a strong hint, not a fact. Combine it with the situation tags (src/tags.py) for confidence.
"""
import csv
import datetime as dt
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "ronaldo.db"
CSV = ROOT / "data" / "matches" / "matches.csv"
WINDOW = 2  # days after a match that still count; searches choose their own tighter limit with --within


def build() -> None:
    con = sqlite3.connect(DB)
    con.executescript(
        """
        DROP TABLE IF EXISTS matches;
        DROP TABLE IF EXISTS post_match;
        CREATE TABLE matches (date TEXT, team TEXT, opponent TEXT, venue TEXT, goals_for INTEGER, goals_against INTEGER,
                              result TEXT, competition TEXT, round TEXT);
        CREATE TABLE post_match (code TEXT PRIMARY KEY, match_date TEXT, team TEXT, opponent TEXT, venue TEXT,
                                 goals_for INTEGER, goals_against INTEGER, result TEXT, competition TEXT, days_after INTEGER);
        """
    )
    matches = []
    with open(CSV, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            matches.append((r["date"], r["team"], r["opponent"], r["venue"], int(r["goals_for"]), int(r["goals_against"]),
                            r["result"], r["competition"], r["round"]))
    con.executemany("INSERT INTO matches VALUES (?,?,?,?,?,?,?,?,?)", matches)
    by_date = {}
    for m in matches:
        by_date.setdefault(dt.date.fromisoformat(m[0]), []).append(m)

    joined = 0
    for code, ts in con.execute("SELECT code, ts FROM posts WHERE ts IS NOT NULL").fetchall():
        post_day = dt.datetime.fromtimestamp(ts, dt.timezone.utc).date()
        for back in range(0, WINDOW + 1):  # nearest match first
            ms = by_date.get(post_day - dt.timedelta(days=back))
            if ms:
                d, team, opp, venue, gf, ga, res, comp, _ = sorted(ms)[-1]
                con.execute("INSERT INTO post_match VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (code, d, team, opp, venue, gf, ga, res, comp, back))
                joined += 1
                break
    con.execute("CREATE INDEX idx_pm_result ON post_match(result, days_after)")
    con.commit()
    print(f"{len(matches)} matches loaded; {joined} of {con.execute('SELECT COUNT(*) FROM posts').fetchone()[0]} posts joined to a match")


def report() -> None:
    con = sqlite3.connect(DB)
    print("by result and days after:")
    for res, d, n in con.execute("SELECT result, days_after, COUNT(*) FROM post_match GROUP BY 1,2 ORDER BY 1,2"):
        print(f"  {res}  +{d}d  {n}")
    print("by team:", con.execute("SELECT team, COUNT(*) FROM post_match GROUP BY 1 ORDER BY 2 DESC").fetchall())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    report() if "--report" in sys.argv else build()
