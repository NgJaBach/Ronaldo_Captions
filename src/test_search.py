"""VAR for the search engine. Run from the repo root:  python src/test_search.py
Relevance is judged by regex on the returned captions: (query, pattern, hits needed, within top N).
Small on purpose: it guards against regressions, it is not a benchmark."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search import Searcher

CASES = [
    ("women's day", r"women", 3, 3),
    ("International Women's Day thank you to the women in my life", r"women", 2, 3),
    ("never give up", r"never give up|never stop|not stopping", 3, 5),
    ("don't ever quit", r"never give up|never stop|not stopping|keep (going|believing)|never settle|never let", 3, 10),  # zero shared words
    ("honouring the females in my family", r"women|mother|sisters|daughters|mum", 1, 10),                               # paraphrase
    ("so thankful for everyone who supports me", r"thank|grateful|gratitude|support", 6, 10),
    ("training hard before the match", r"train|work|gym|workout|recovery|practice|preparation", 6, 10),
    ("proud of my team after the win", r"proud|win|victory|champion|team|trophy|title", 6, 10),
]

s = Searcher()
failed = []
for q, rx, need, n in CASES:
    caps = [r["caption"] for r in s.search(q, k=n)]
    hits = sum(bool(re.search(rx, c, re.I)) for c in caps)
    ok = hits >= need
    print(f"{'ok  ' if ok else 'FAIL'} {hits}/{need} in top {n}: {q!r}")
    if not ok:
        failed.append(q)

# Determinism: same query twice, identical ranking.
a = [r["code"] for r in s.search("grateful", k=50)]
assert a == [r["code"] for r in s.search("grateful", k=50)], "ranking is not deterministic"

# Paging: pages must tile the full ranking with no overlap or gaps.
p1 = [r["code"] for r in s.search("family", k=10, offset=0)]
p2 = [r["code"] for r in s.search("family", k=10, offset=10)]
assert p1 + p2 == [r["code"] for r in s.search("family", k=20)], "paging does not tile"

# k=None returns the whole ranking (the --all path).
assert len(s.search("family", k=None)) > 1000, "full ranking is too short"

# Filters: short aura lines only; collab posts excluded by default.
assert all(len(r["caption"]) <= 40 for r in s.search("work", k=20, max_len=40)), "max_len filter leaked"
owners = {s.con.execute("select owner from posts where code=?", (r["code"],)).fetchone()[0] for r in s.search("brand", k=50)}
assert owners <= {"cristiano"}, f"collab posts leaked: {owners}"

# Moods: a bare feeling word is searched by phrases in his voice, not by the word.
SETBACK_RX = r"result we wanted|come back stronger|setback|not stopping|keep fighting|rise|fight isn.t over|difficult"
sad = s.search("sad", k=10)
assert sum(bool(re.search(SETBACK_RX, r["caption"], re.I)) for r in sad) >= 4, "mood 'sad' did not find setback captions"
assert [r["code"] for r in sad] != [r["code"] for r in s.search("sad", k=10, expand=False)], "expand=False changed nothing"
assert s.mood_for("Sad!") == "sad" and s.mood_for("how sad") is None, "mood trigger rules changed"
try:
    s.ranking("x", mood="nonsense")
    raise SystemExit("unknown mood should raise")
except ValueError:
    pass

# Context filters: "after a loss" must really be after a loss, and agree with the sad mood.
after_loss = s.search("sad", k=8, after="loss", no_ads=True)
assert after_loss and all(r["context"] and r["context"]["result"] == "L" and r["context"]["days_after"] <= 1 for r in after_loss), "after=loss leaked"
assert sum(bool(re.search(SETBACK_RX, r["caption"], re.I)) for r in after_loss) >= 4, "sad + after loss lost its precision"
for r in s.search("work", k=20, after="win", team="Al Nassr"):
    assert r["context"]["result"] == "W" and r["context"]["team"] == "Al Nassr", "team/after filter leaked"
assert all("sponsor" not in r["tags"] for r in s.search("download app", k=30, no_ads=True)), "no_ads leaked a sponsor post"
assert all("setback" in r["tags"] for r in s.search("result", k=15, tag="setback")), "tag filter leaked"
try:
    s.ranking("x", after="sideways")
    raise SystemExit("bad `after` should raise")
except ValueError:
    pass
w = s.search("work", k=1, within=0, after="win")
assert all(r["context"]["days_after"] == 0 for r in w), "within=0 must mean same day only"

# Emoji-only captions are hidden by default but retrievable on request.
assert all(sum(ch.isalpha() for ch in r["caption"]) >= 3 for r in s.search("legacy", k=30)), "emoji-only leaked"
assert any(sum(ch.isalpha() for ch in r["caption"]) < 3 for r in s.search("legacy", k=None, emoji_only=True)), "emoji_only flag broken"

print("determinism, paging, full ranking and filters ok")
sys.exit(1 if failed else 0)
