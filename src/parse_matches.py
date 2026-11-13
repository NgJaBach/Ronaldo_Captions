"""The statistician. Turns the downloaded Wikipedia wikitext into data/matches/matches.csv.

  python src/parse_matches.py

One row per match of the team Ronaldo played for at the time (see TEAMS), as that team saw it:
date, team, opponent, venue (H/A/N/?), goals_for, goals_against, result (W/D/L), competition, round, source.
Two page layouts exist on Wikipedia: {{football box}} templates (most pages) and plain result tables (Manchester United).
Source text: Wikipedia, CC BY-SA 4.0 (see data/matches/NOTICE.md).
"""
import csv
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "matches" / "raw"
OUT = ROOT / "data" / "matches" / "matches.csv"

# team label -> substrings that identify it inside a cleaned team1/team2 string (lower case)
TEAMS = {
    "Real Madrid": ["real madrid"],
    "Juventus": ["juventus"],
    "Manchester United": ["manchester united"],
    "Al Nassr": ["al-nassr", "al nassr"],
    "Portugal": ["portugal", "por"],  # the Portugal pages write the side as {{fb-rt|POR}}, which cleans to "POR"
}
# Matches count only while Ronaldo was with the team (to the day, from the public transfer announcements).
# A post from October 2022 must not be joined to an Al Nassr match he had not yet signed for.
TENURE = {
    "Real Madrid": (dt.date(2017, 1, 1), dt.date(2018, 7, 9)),
    "Juventus": (dt.date(2018, 7, 10), dt.date(2021, 8, 26)),
    "Manchester United": (dt.date(2021, 8, 27), dt.date(2022, 11, 22)),
    "Al Nassr": (dt.date(2022, 12, 30), dt.date(2099, 1, 1)),
    "Portugal": (dt.date(2017, 1, 1), dt.date(2099, 1, 1)),
}
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)}


def strip_refs(s: str) -> str:
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"<ref[^>]*?/>", "", s)
    s = re.sub(r"<ref[^>]*?>.*?</ref>", "", s, flags=re.S)
    return s


def clean(s: str) -> str:
    s = strip_refs(s)
    s = re.sub(r"\{\{\s*fb(?:-rt)?\s*\|\s*([^{}|]+?)\s*\}\}", r"\1", s)  # {{fb|CRO}} -> CRO, {{fb-rt|Portugal}} -> Portugal
    for _ in range(4):  # drop remaining templates, innermost first
        s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", s)
    s = re.sub(r"\[https?://\S+\]", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("&nbsp;", " ").replace("\xa0", " ").replace("'''", "").replace("''", "")
    return re.sub(r"\s+", " ", s).strip()


def parse_date(raw: str):
    raw = strip_refs(raw)
    m = re.search(r"[Ss]tart date[^}]*?(\d{4})\s*\|\s*(\d{1,2})\s*\|\s*(\d{1,2})", raw)
    if m:
        return dt.date(int(m[1]), int(m[2]), int(m[3]))
    t = clean(raw)
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", t)
    if m and m[2].lower() in MONTHS:
        return dt.date(int(m[3]), MONTHS[m[2].lower()], int(m[1]))
    m = re.search(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", t)
    if m and m[1].lower() in MONTHS:
        return dt.date(int(m[3]), MONTHS[m[1].lower()], int(m[2]))
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        return dt.date(int(m[1]), int(m[2]), int(m[3]))
    return None


def parse_score(s: str):
    m = re.search(r"(\d+)\s*[–-]\s*(\d+)", clean(s))
    return (int(m[1]), int(m[2])) if m else None


def headings(text: str):
    return [(m.start(), clean(m[2])) for m in re.finditer(r"^(={2,6})\s*(.+?)\s*\1\s*$", text, flags=re.M)]


def heading_at(hs, pos):
    cur = ""
    for p, h in hs:
        if p > pos:
            break
        cur = h
    return cur


def split_top(body: str):
    """Split a template body on '|' at brace/bracket depth 0."""
    parts, depth, cur, i = [], 0, [], 0
    while i < len(body):
        two = body[i:i + 2]
        if two in ("{{", "[["):
            depth += 1; cur.append(two); i += 2; continue
        if two in ("}}", "]]"):
            depth -= 1; cur.append(two); i += 2; continue
        if body[i] == "|" and depth == 0:
            parts.append("".join(cur)); cur = []
        else:
            cur.append(body[i])
        i += 1
    parts.append("".join(cur))
    return parts


def football_boxes(text: str):
    for m in re.finditer(r"\{\{\s*[Ff]ootball ?box(?: collapsible)?\s*\n?", text):
        depth, i = 1, m.end()  # the opening "{{" is one level; the matching "}}" closes the box
        while i < len(text) and depth > 0:
            two = text[i:i + 2]
            if two == "{{":
                depth += 1; i += 2
            elif two == "}}":
                depth -= 1; i += 2
            else:
                i += 1
        fields = {}
        for part in split_top(text[m.end(): i - 2]):
            if "=" in part:
                k, v = part.split("=", 1)
                fields[k.strip().lower()] = v
        yield m.start(), fields


def side(ours: str, name1: str, name2: str):
    keys = TEAMS[ours]
    a = any(k in name1.lower() for k in keys)
    b = any(k in name2.lower() for k in keys)
    if ours == "Portugal":  # avoid matching "por" inside unrelated names
        a = name1.strip().upper() == "POR" or "portugal" in name1.lower()
        b = name2.strip().upper() == "POR" or "portugal" in name2.lower()
    return 1 if a and not b else 2 if b and not a else 0


def rows_from_boxes(team: str, text: str, src: str):
    hs = headings(text)
    for pos, f in football_boxes(text):
        d, sc = parse_date(f.get("date", "")), parse_score(f.get("score", ""))
        n1, n2 = clean(f.get("team1", "")), clean(f.get("team2", ""))
        who = side(team, n1, n2)
        if not d or not sc or not who:
            continue
        gf, ga = (sc[0], sc[1]) if who == 1 else (sc[1], sc[0])
        yield dict(date=d, team=team, opponent=(n2 if who == 1 else n1), venue="H" if who == 1 else "A",
                   gf=gf, ga=ga, competition=heading_at(hs, pos), round=clean(f.get("round", "")), source=src,
                   stated=clean(f.get("result", "")))


def rows_from_tables(team: str, text: str, src: str):
    hs = headings(text)
    starts = list(re.finditer(r"^\|-.*$", text, flags=re.M))
    for k, m in enumerate(starts):
        end = starts[k + 1].start() if k + 1 < len(starts) else len(text)
        body = text[m.end():end]
        cells = []
        for line in body.split("\n"):
            if line.startswith("|}"):
                break  # the table ended
            if not line.startswith("|") or line.startswith("|-"):
                continue
            for c in line[1:].split("||"):
                cells.append(c)
        ha = next((i for i, c in enumerate(cells) if clean(c) in ("H", "A", "N")), None)
        di = next((i for i, c in enumerate(cells) if parse_date(c)), None)
        if ha is None or di is None or ha + 1 >= len(cells) or ha < 1:
            continue
        sc = parse_score(cells[ha + 1])
        if not sc:
            continue
        yield dict(date=parse_date(cells[di]), team=team, opponent=clean(cells[ha - 1]), venue=clean(cells[ha]),
                   gf=sc[0], ga=sc[1], competition=heading_at(hs, m.start()), round="", source=src, stated="")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    for wiki in sorted(RAW.glob("*.wiki")):
        meta = json.loads(wiki.with_suffix(".meta.json").read_text(encoding="utf-8"))
        text, team = wiki.read_text(encoding="utf-8"), meta["team"]
        got = list(rows_from_boxes(team, text, meta["title"])) + list(rows_from_tables(team, text, meta["title"]))
        print(f"{len(got):4} matches  {meta['title']}")
        rows += got
    # de-duplicate (same team, date, opponent), prefer the row that states its own result
    best = {}
    for r in rows:
        k = (r["date"], r["team"], r["opponent"].lower())
        if k not in best or (r["stated"] and not best[k]["stated"]):
            best[k] = r
    out = [r for r in best.values() if TENURE[r["team"]][0] <= r["date"] <= TENURE[r["team"]][1]]
    print(f"{len(best) - len(out)} matches outside Ronaldo's tenure with the team dropped")
    out.sort(key=lambda r: (r["date"], r["team"]))
    # self-check: where a page states its own W/D/L, it must agree with the score we parsed
    checked = bad = 0
    for r in out:
        if r["stated"] in ("W", "D", "L"):
            checked += 1
            mine = "W" if r["gf"] > r["ga"] else "L" if r["gf"] < r["ga"] else "D"
            if mine != r["stated"]:
                bad += 1
                print("  DISAGREES:", r["date"], r["team"], r["opponent"], r["gf"], r["ga"], "page says", r["stated"])
    print(f"result self-check: {checked} matches state their own result, {bad} disagree")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "team", "opponent", "venue", "goals_for", "goals_against", "result", "competition", "round", "source"])
        for r in out:
            res = "W" if r["gf"] > r["ga"] else "L" if r["gf"] < r["ga"] else "D"
            w.writerow([r["date"].isoformat(), r["team"], r["opponent"], r["venue"], r["gf"], r["ga"], res,
                        r["competition"], r["round"], r["source"]])
    print(f"wrote {len(out)} matches to {OUT.name} ({len(rows) - len(best)} duplicates dropped)")


if __name__ == "__main__":
    main()
