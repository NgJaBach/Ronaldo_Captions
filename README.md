# 🐐 Ronaldo Captions

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Cristiano_Ronaldo_0876.jpg/960px-Cristiano_Ronaldo_0876.jpg" alt="Cristiano Ronaldo celebrating a goal" width="600">
  <br>
  <sub>Photo: <a href="https://commons.wikimedia.org/wiki/File:Cristiano_Ronaldo_0876.jpg">YantsImages / Aram Nersesyan</a>, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a></sub>
</p>

> 4,145 posts. 2,412 captions. 1 dataset. Zero chill.

Every caption from [@cristiano](https://www.instagram.com/cristiano/)'s Instagram, in one searchable SQLite file,
so you can study the master's writing without scrolling for 4,000 posts. Because we have lives. Mostly.

**Why?** To learn how Ronaldo writes, and to build search / RAG / style tools on top. Pure fan project. Not affiliated with Ronaldo or Meta.

## How we did it

1. 🕵️ Opened his profile in a logged-in browser and scrolled slowly, like a human who is *definitely* not a robot.
2. 📝 A tiny script quietly wrote down the posts Instagram was already sending to the page. No hammering, no tricks.
3. 📦 Dropped everything into SQLite with full-text search. Nothing filtered, nothing judged.

## What you get

- **2,412 of 4,145 posts**, March 2017 → October 2026. Instagram stops loading older ones, so the 2012–2017 era is missing. We tried. Even legends have limits.
- Caption, date, likes, comments, post type and who posted it.
- Emoji-only posts, Portuguese, Arabic and collab posts are all in there. Filter them yourself. 🇵🇹

## Search it (no chatbot needed) 🔎

Ask in plain words. It finds captions by **meaning**, not just matching words, runs on a normal CPU, and gives the same answer every time.

```bash
pip install -r requirements.txt
python src/search.py "women's day"                          # top 10
python src/search.py "don't ever quit" -k 25                # finds "We're not stopping", "Never give up..."
python src/search.py "confidence" --max-len 40              # short aura lines only
python src/search.py "family" --all --offset 50 -k 50       # page through the entire ranking
python src/search.py -i                                     # stay open: slow start once, instant after
```

### "Sad", but the way *he* says it 😤

Ronaldo almost never writes "sad" (18 captions in 2,349 use setback words). After a defeat he writes "We will come back stronger."
So we gave the search some context: **moods** (a feeling word is searched by phrases in his voice), **tags** (setback, win, family...),
and **match results** (each post is joined to the match it followed, from Wikipedia), so you can ask for posts *after a loss*:

```bash
python src/search.py sad --after loss --no-ads              # his real reactions to defeats, with the match shown
python src/search.py "back to work" --team "Al Nassr" --after win
python src/search.py --mood motivated --tag training
```

Prefer a page to a terminal? `python src/serve.py` and open the address it prints (usually <http://127.0.0.1:8000>). It has mood chips, an
After win/draw/loss filter, team and tag filters, a Copy button, and a SIUUU button. 🔊
How we know it works and where it doesn't: [docs/context.md](docs/context.md).

How: a small multilingual embedding model (`multilingual-e5-small`) turns each caption into numbers, those get fused with
classic keyword search (SQLite BM25) using Reciprocal Rank Fusion. Vectors are precomputed, so nothing changes between
queries. Starting Python takes about a minute (PyTorch is a slow walker), then each query takes milliseconds, so use `-i`.

```text
.
├── data/    the loot (ronaldo.db, embeddings, matches/ from Wikipedia, raw JSONL)
├── design/  the stadium's front door (the HTML interface)
├── docs/    how the context features work and how they were checked
└── src/     capture.js (scout), receiver.py (ball boy), ronaldo_db.py (vault), embed.py (academy),
             search.py (striker), serve.py (stadium), fetch_matches.py + parse_matches.py + context.py
             (statisticians), tags.py (tagger), moods.json, test_*.py (VAR)
```

## Fair warning 🟨

Match results in `data/matches/` come from Wikipedia (CC BY-SA 4.0, see the notice there).
The captions belong to Ronaldo and his team, and Instagram's Terms of Use don't allow automated collection.
This is shared for research and fun. If you are a rights holder and want something removed, open an issue and it's gone.
If you copy the scraper, be gentle: slow scrolling, no parallel runs, stop on errors.

*Siuuuu.*
