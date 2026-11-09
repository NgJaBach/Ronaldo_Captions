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

# Emoji-only captions are hidden by default but retrievable on request.
assert all(sum(ch.isalpha() for ch in r["caption"]) >= 3 for r in s.search("legacy", k=30)), "emoji-only leaked"
assert any(sum(ch.isalpha() for ch in r["caption"]) < 3 for r in s.search("legacy", k=None, emoji_only=True)), "emoji_only flag broken"

print("determinism, paging, full ranking and filters ok")
sys.exit(1 if failed else 0)
