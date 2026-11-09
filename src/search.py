"""The striker. Ask in plain words, get his captions back, best first. No chatbot required.

Usage (repo root):
  python src/search.py "women's day"                  # top 10
  python src/search.py "never give up" -k 25
  python src/search.py "gratitude to family" --all --offset 50 -k 50   # page through the full ranking
  python src/search.py "work mode" --max-len 40 --since 2022-01-01     # short aura lines only
  python src/search.py "legacy" --mode keyword|semantic|hybrid         # hybrid is the default
  python src/search.py "legacy" --json                                  # machine-readable, for building on top
  python src/search.py -i                                               # stay open: one slow start, then instant answers

How: caption vectors (src/embed.py) are compared with the query vector (meaning), the SQLite FTS5 index finds exact
words (BM25), and the two rankings are fused with Reciprocal Rank Fusion (Cormack et al., 2009). Everything is
precomputed and deterministic: the same query gives the same ranking until the data changes.

Python use:  from search import Searcher;  Searcher().search("women's day", k=5)
"""
import argparse
import datetime as dt
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
RRF_K = 60  # standard constant from the RRF paper
KW_WEIGHT = 0.3  # keyword vote weight in the fusion (semantic is 1.0). Plateau of 8/9 test cases from 0.15 to 0.6;
#                  0.3 keeps exact-word queries solid while letting meaning lead on conversational ones.
# Words too common to be worth a keyword hit (the semantic half already understands them).
STOP = {"the", "and", "for", "with", "that", "this", "you", "your", "are", "was", "has", "have", "not", "but",
        "day", "his", "her", "our", "all", "from", "about", "into", "what", "when", "who"}


class Searcher:
    def __init__(self):
        meta = json.loads(META.read_text(encoding="utf-8"))
        self.model_name = meta["model"]
        self.codes = [r["code"] for r in meta["rows"]]
        self.vecs = np.load(NPY).astype(np.float32)
        self.con = sqlite3.connect(DB, check_same_thread=False)  # serve.py shares one Searcher across threads (with a lock)
        self.con.row_factory = sqlite3.Row
        self._model = None  # loaded lazily: keyword-only searches stay instant
        # Emoji-only captions ("🙏🏽", "🏆") look mildly similar to every query in embedding space, so they are
        # hidden unless asked for. They stay in the database; this is a search-time filter only.
        self.texty = {c for c, cap in self.con.execute("SELECT code, caption FROM posts")
                      if sum(ch.isalpha() for ch in cap) >= 3}

    def _embed_query(self, q: str) -> np.ndarray:
        if self._model is None:
            import os
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            import transformers
            from sentence_transformers import SentenceTransformer
            transformers.logging.set_verbosity_error()
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model.encode(["query: " + q], normalize_embeddings=True)[0]

    def _allowed(self, owner, since, until, max_len, min_len, min_likes, emoji_only=False):
        """Set of codes that pass the filters (None = no filter active)."""
        where, args = ["caption != ''"], []
        if owner:
            where.append("owner = ?"); args.append(owner)
        if since:
            where.append("ts >= ?"); args.append(int(dt.datetime.fromisoformat(since).replace(tzinfo=dt.timezone.utc).timestamp()))
        if until:
            where.append("ts < ?"); args.append(int(dt.datetime.fromisoformat(until).replace(tzinfo=dt.timezone.utc).timestamp()))
        if max_len:
            where.append("length(caption) <= ?"); args.append(max_len)
        if min_len:
            where.append("length(caption) >= ?"); args.append(min_len)
        if min_likes:
            where.append("likes >= ?"); args.append(min_likes)
        codes = {r[0] for r in self.con.execute("SELECT code FROM posts WHERE " + " AND ".join(where), args)}
        return codes if emoji_only else codes & self.texty

    def _keyword_ranking(self, q: str, allowed: set) -> list:
        words = [w for w in re.findall(r"\w+", q, flags=re.UNICODE) if len(w) > 2 and w.lower() not in STOP]
        if not words:
            return []
        match = " OR ".join(f'"{w}"' for w in words)
        rows = self.con.execute(
            "SELECT p.code FROM posts_fts f JOIN posts p ON p.rowid = f.rowid "
            "WHERE posts_fts MATCH ? ORDER BY rank LIMIT 2000", (match,)).fetchall()
        return [r[0] for r in rows if r[0] in allowed]

    def _semantic_ranking(self, q: str, allowed: set) -> list:
        sims = self.vecs @ self._embed_query(q)
        order = np.argsort(-sims, kind="stable")
        return [(self.codes[i], float(sims[i])) for i in order if self.codes[i] in allowed]

    def ranking(self, q, mode="hybrid", owner="cristiano", since=None, until=None,
                max_len=None, min_len=None, min_likes=None, kw_weight=KW_WEIGHT, emoji_only=False):
        """Full ranked list of (code, score). Cheap enough to compute for all ~2.4k posts."""
        allowed = self._allowed(owner, since, until, max_len, min_len, min_likes, emoji_only)
        kw = self._keyword_ranking(q, allowed) if mode in ("keyword", "hybrid") else []
        sem = self._semantic_ranking(q, allowed) if mode in ("semantic", "hybrid") else []
        if mode == "keyword":
            return [(c, 1.0 / (RRF_K + i + 1)) for i, c in enumerate(kw)]
        if mode == "semantic":
            return sem
        score = {}
        for i, (c, _) in enumerate(sem):
            score[c] = score.get(c, 0.0) + 1.0 / (RRF_K + i + 1)
        for i, c in enumerate(kw):
            score[c] = score.get(c, 0.0) + kw_weight / (RRF_K + i + 1)
        return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))  # code breaks ties: fully deterministic

    def search(self, q, k=10, offset=0, **filters):
        return self.page(self.ranking(q, **filters), k, offset)

    def page(self, ranked, k=10, offset=0):
        """Slice a ranking (from ranking()) into a page of full result dicts."""
        page = ranked if k is None else ranked[offset: offset + k]
        out = []
        for rank, (code, score) in enumerate(page, start=offset + 1):
            r = self.con.execute("SELECT code, ts, likes, comments, type, caption FROM posts WHERE code=?", (code,)).fetchone()
            out.append({"rank": rank, "score": round(score, 4), "code": r["code"],
                        "date": dt.datetime.fromtimestamp(r["ts"], dt.timezone.utc).strftime("%Y-%m-%d"),
                        "likes": r["likes"], "type": r["type"], "caption": r["caption"],
                        "url": f"https://www.instagram.com/p/{r['code']}/"})
        return out


def show(res) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    for r in res:
        cap = r["caption"].replace("\n", " ⏎ ")
        print(f"{r['rank']:>3}. [{r['date']}] {r['likes']/1e6:5.1f}M  {cap[:220]}{'…' if len(cap) > 220 else ''}")
        print(f"     {r['url']}  (score {r['score']})")
    if not res:
        print("No results. Even Ronaldo has off days.")


def repl() -> None:
    """Load once (about a minute of imports), then answer in milliseconds. Type :help for commands."""
    s = Searcher()
    s._embed_query("warm up")
    k, mode, last, offset = 10, "hybrid", None, 0
    print("Locker room is open. Type a query, :more for the next page, :k N, :mode hybrid|semantic|keyword, :quit")
    while True:
        try:
            line = input("\nsearch> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in (":q", ":quit", ":exit"):
            break
        if line.startswith(":k "):
            k = int(line[3:]); print(f"k = {k}"); continue
        if line.startswith(":mode "):
            mode = line[6:].strip(); print(f"mode = {mode}"); continue
        if line == ":more":
            if last is None:
                print("Nothing to continue yet."); continue
            offset += k
        elif line.startswith(":"):
            print("Commands: :more  :k N  :mode hybrid|semantic|keyword  :quit"); continue
        else:
            last, offset = line, 0
        show(s.search(last, k=k, offset=offset, mode=mode))


def main() -> None:
    ap = argparse.ArgumentParser(description="Semantic + keyword search over Ronaldo's captions")
    ap.add_argument("query", nargs="?")
    ap.add_argument("-i", "--interactive", action="store_true", help="stay open and answer many queries fast")
    ap.add_argument("-k", type=int, default=10, help="how many results (default 10)")
    ap.add_argument("--all", action="store_true", help="return the full ranking (combine with --offset/-k to page)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--mode", choices=["hybrid", "semantic", "keyword"], default="hybrid")
    ap.add_argument("--owner", default="cristiano", help="'' to include collab posts by other accounts")
    ap.add_argument("--since"), ap.add_argument("--until")
    ap.add_argument("--max-len", type=int), ap.add_argument("--min-len", type=int), ap.add_argument("--min-likes", type=int)
    ap.add_argument("--emoji-only", action="store_true", help="also return captions with no words (hidden by default)")
    ap.add_argument("--kw-weight", type=float, default=KW_WEIGHT, help="keyword vote weight in hybrid (default %(default)s)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.interactive:
        return repl()
    if not a.query:
        ap.error("give a query, or use -i for interactive mode")

    s = Searcher()
    k = None if (a.all and a.offset == 0 and a.k == 10) else a.k
    res = s.search(a.query, k=k, offset=a.offset, mode=a.mode, owner=a.owner or None, since=a.since,
                   until=a.until, max_len=a.max_len, min_len=a.min_len, min_likes=a.min_likes, kw_weight=a.kw_weight,
                   emoji_only=a.emoji_only)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    show(res)


if __name__ == "__main__":
    main()
