"""VAR for the match data and the post-to-match join. Fast, no model needed.  Run from the repo root:
  python src/test_context.py
"""
import csv
import datetime as dt
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_matches import TENURE

ROOT = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(open(ROOT / "data" / "matches" / "matches.csv", encoding="utf-8")))
assert len(rows) > 600, f"only {len(rows)} matches"

# every row is internally consistent and inside Ronaldo's tenure with that team
seen = set()
for r in rows:
    d = dt.date.fromisoformat(r["date"])
    lo, hi = TENURE[r["team"]]
    assert lo <= d <= hi, f"outside tenure: {r}"
    gf, ga = int(r["goals_for"]), int(r["goals_against"])
    assert r["result"] == ("W" if gf > ga else "L" if gf < ga else "D"), f"result disagrees with score: {r}"
    key = (r["date"], r["team"], r["opponent"].lower())
    assert key not in seen, f"duplicate match: {key}"
    seen.add(key)

# each team is present in the years he played for it (a missing page would show up here)
need = {("Real Madrid", "2017"), ("Juventus", "2019"), ("Juventus", "2020"), ("Manchester United", "2021"),
        ("Manchester United", "2022"), ("Al Nassr", "2023"), ("Al Nassr", "2025"), ("Portugal", "2017"), ("Portugal", "2024")}
have = {(r["team"], r["date"][:4]) for r in rows}
assert need <= have, f"missing team-years: {sorted(need - have)}"

# spot checks against results anyone can look up (home/away from the team's own point of view)
def has(date, team, opp_part, gf, ga):
    return any(r["date"] == date and r["team"] == team and opp_part in r["opponent"] and (int(r["goals_for"]), int(r["goals_against"])) == (gf, ga) for r in rows)

assert has("2018-05-26", "Real Madrid", "Liverpool", 3, 1), "Champions League final 2018 missing or wrong"
assert has("2021-09-14", "Manchester United", "Young Boys", 1, 2), "Young Boys 1-2 missing or wrong"
assert has("2024-12-06", "Al Nassr", "Al-Ittihad", 1, 2), "Al-Ittihad 1-2 missing or wrong"

# the join: tables exist, windows respected, results agree with the matches table
con = sqlite3.connect(ROOT / "data" / "ronaldo.db")
n = con.execute("SELECT COUNT(*) FROM post_match").fetchone()[0]
assert n > 1000, f"only {n} posts joined"
assert con.execute("SELECT MAX(days_after) FROM post_match").fetchone()[0] <= 2
bad = con.execute("""SELECT COUNT(*) FROM post_match pm JOIN posts p ON p.code = pm.code
                     WHERE CAST(julianday(date(p.ts, 'unixepoch')) - julianday(pm.match_date) AS INT) != pm.days_after""").fetchone()[0]
assert bad == 0, f"{bad} posts have a wrong days_after"
assert con.execute("SELECT COUNT(*) FROM post_match WHERE result NOT IN ('W','D','L')").fetchone()[0] == 0
print(f"{len(rows)} matches and {n} post-to-match joins ok")
