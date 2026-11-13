"""The match-day programme. Downloads match-result wikitext from Wikipedia (CC BY-SA 4.0) for the teams Ronaldo played for.

  python src/fetch_matches.py --probe   # which pages exist (1 request)
  python src/fetch_matches.py           # download missing pages to data/matches/raw/ (about 15 requests, 1.5 s apart)

Polite by design: a descriptive User-Agent, a pause between requests, and pages already on disk are never fetched again.
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "matches" / "raw"
API = "https://en.wikipedia.org/w/api.php"
UA = "ronaldo-captions-research/0.1 (personal research project; https://github.com/NgJaBach/Ronaldo)"

# (team label, page title). Posts run March 2017 to October 2026.
PAGES = [
    ("Real Madrid", "2016–17 Real Madrid CF season"),
    ("Real Madrid", "2017–18 Real Madrid CF season"),
    ("Juventus", "2018–19 Juventus FC season"),
    ("Juventus", "2019–20 Juventus FC season"),
    ("Juventus", "2020–21 Juventus FC season"),
    ("Manchester United", "2021–22 Manchester United F.C. season"),
    ("Manchester United", "2022–23 Manchester United F.C. season"),
    # Manchester United's group-stage games are transcluded from these pages, not in their own season pages.
    ("Manchester United", "2021–22 UEFA Champions League group stage"),
    ("Manchester United", "2022–23 UEFA Europa League group stage"),
    ("Al Nassr", "2022–23 Al-Nassr FC season"),
    ("Al Nassr", "2023–24 Al-Nassr FC season"),
    ("Al Nassr", "2024–25 Al-Nassr FC season"),
    ("Al Nassr", "2025–26 Al-Nassr FC season"),
    ("Al Nassr", "2026–27 Al-Nassr FC season"),
    ("Portugal", "Portugal national football team results (2000–2019)"),
    ("Portugal", "Portugal national football team results (2020–present)"),
]


def api(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")


def probe() -> None:
    titles = [t for _, t in PAGES]
    q = api({"action": "query", "titles": "|".join(titles), "redirects": "1"})["query"]
    redirects = {r["from"]: r["to"] for r in q.get("redirects", [])}
    norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
    found = {p["title"]: not p.get("missing", False) for p in q["pages"]}
    for team, t in PAGES:
        resolved = redirects.get(norm.get(t, t), norm.get(t, t))
        mark = "ok     " if found.get(resolved) else "MISSING"
        print(f"{mark} {team:18} {t}" + (f"  -> {resolved}" if resolved != t else ""))


def fetch() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for team, t in PAGES:
        out = RAW / f"{slug(t)}.wiki"
        if out.exists():
            print(f"have    {t}")
            continue
        time.sleep(1.5)
        d = api({"action": "query", "prop": "revisions", "rvprop": "content|ids", "rvslots": "main", "titles": t, "redirects": "1"})
        pages = d["query"]["pages"]
        if not pages or pages[0].get("missing"):
            print(f"MISSING {t}")
            continue
        rev = pages[0]["revisions"][0]
        out.write_text(rev["slots"]["main"]["content"], encoding="utf-8")
        (RAW / f"{slug(t)}.meta.json").write_text(json.dumps(
            {"team": team, "title": pages[0]["title"], "revid": rev["revid"], "license": "CC BY-SA 4.0",
             "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(pages[0]["title"].replace(" ", "_"))},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"saved   {t}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    probe() if a.probe else fetch()
