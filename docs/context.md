# Context search: moods, tags, and "after a loss"

Searching "sad" over Ronaldo's captions returned "Mood" and "Buongiorno", because he almost never writes the word
(only 18 of 2,349 captions use explicit setback words). The context lives in the world, not the caption. Three features
fix that. Rebuild order: `fetch_matches.py` > `parse_matches.py` > `context.py` > `tags.py`.

## 1. Moods (`src/moods.json`)
A one-word feeling is searched by the average meaning of phrases in his voice instead of the bare word.
Checked by reading the top 4 results per mood: sad, motivated, love, grateful, proud, confident and ambitious are
convincing; happy and tired are fair; nostalgic and calm are weak. A "humble" mood returned noise and was removed.

## 2. Tags (`src/tags.py`)
Zero-shot: each tag is a few example phrases; a caption's score is its best cosine to them, converted to a z-score across
captions with real words (the model's raw cosines are all 0.8+, so raw thresholds tagged half the corpus). A tag needs
z >= 2.0 (win 1.8, injury 2.5, tribute 2.5). `sponsor` is rule-based (links, brand handles, "download").
Validation, independent of the tagger: the match data says which posts followed a loss or a win.

| z >= 2.0 | after a loss | after a draw | after a win | no match |
|---|---|---|---|---|
| `setback` tag | 14.0% | 4.3% | 1.2% | 0.6% |
| `win` tag | 0.0% | 0.7% | 3.9% | 0.3% |

Recall is modest (most post-loss posts are ads, family or emoji), and tags such as win, trophy, fans and gratitude overlap.

## 3. Match context (`src/fetch_matches.py`, `parse_matches.py`, `context.py`)
Source: 16 Wikipedia pages (season pages for Real Madrid, Juventus, Manchester United, Al-Nassr; Portugal results lists;
two group-stage pages that Manchester United's pages transclude). 689 matches, counted only within his tenure with each team.
- Parser self-check: 615 matches state their own result on the page; all 615 agree with the score parsed (0 disagree).
- Spot checks of known fixtures (two Champions League finals, a knockout second leg, Young Boys 1-2, Al-Ittihad 1-2) match.
  Every match was not checked one by one.
- Join: a post is "after" the most recent match from the same day up to 2 days earlier (UTC). 1,379 of 2,412 posts join.

Limits: the match date is local and the post time is UTC; kickoff times are not used, so a morning post on match day can
be mislabelled; a post in the window may be about something else (an ad, a family photo); the match is the team's, so it
includes games he did not play; one case in the data is a defeat by Roma that fell on the day he celebrated a title.
Tenure dates (Juventus from 2018-07-10, Manchester United from 2021-08-27, Al Nassr from 2022-12-30, and so on) come from
public transfer announcements and are applied to the day.
