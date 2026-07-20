"""Attach outcomes to predictions, and resolve the canonical game id.

Two jobs, one pass over the MLB StatsAPI schedule, because they need the same
data and it is rude to ask twice:

  1. GRADING. Fantasy grades its own rows and ships them in dip.json, so those
     are taken as given. mlb-edge does not grade anything — its README is
     explicit that edges are "hypotheses until validated by logged CLV" — so
     DIP grades them itself from final boxscores.

  2. EVENT RESOLUTION. Neither source carries a real game id. The gamePk that
     StatsAPI already hands over during grading is the authoritative key, so it
     is captured into entity_events, where cross-source correlation joins.

The scoring rule is lifted from Fantasy's grade.py:result_of rather than
reinvented. In particular a stat landing exactly on a whole-numbered line is a
PUSH ('X'), not a loss: there is no side of the line to have been on. Upstream
excludes those from hit rates, and a second implementation that quietly scored
them 0 would make DIP's accuracy numbers disagree with the source's for reasons
nobody could find.
"""
from __future__ import annotations

import json
import urllib.request

from .contract import norm_entity
from . import store

SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
BOXSCORE = "https://statsapi.mlb.com/api/v1/game/{pk}/boxscore"

# market -> (boxscore stat group, field). Mirrors pick6/grade.py:_STAT.
_STAT = {
    "strikeouts":  ("pitching", "strikeOuts"),
    "hits":        ("batting", "hits"),
    "total_bases": ("batting", "totalBases"),
    "home_runs":   ("batting", "homeRuns"),
    "rbi":         ("batting", "rbi"),
    "runs":        ("batting", "runs"),
}


def _get(url: str, timeout: float = 60.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def result_of(side: str, line: float, actual: float) -> str:
    """'1' correct / '0' incorrect / 'X' push. Same rule as grade.py:result_of.

    Both vocabularies are accepted because the sources disagree on wording:
    Fantasy says more/less, mlb-edge says over/under. They mean the same thing,
    and normalizing here beats making every caller remember which is which.
    """
    if actual == line and float(line).is_integer():
        return "X"
    high = side in ("more", "over")
    correct = actual > line if high else actual < line
    return "1" if correct else "0"


def final_stats(date: str, fetch=_get) -> tuple[dict, dict]:
    """One pass over the day's FINAL games.

    Returns (stats, events):
      stats  norm(player) -> {market: actual}
      events norm(player) -> "mlb:<gamePk>"

    Only Final games are read. An in-progress game has partial stats that would
    grade as a confident loss on every over — which is why the status check is
    the first thing here and not an afterthought.
    """
    sched = fetch(SCHEDULE.format(date=date))
    stats: dict[str, dict[str, float]] = {}
    events: dict[str, str] = {}

    for d in sched.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            pk = g.get("gamePk")
            try:
                box = fetch(BOXSCORE.format(pk=pk))
            except Exception:
                # One unreadable boxscore must not cost the whole day's grading.
                continue
            for team_side in ("home", "away"):
                for pdata in box["teams"][team_side]["players"].values():
                    key = norm_entity(pdata["person"]["fullName"])
                    events[key] = f"mlb:{pk}"
                    st = pdata.get("stats", {})
                    rec = {m: float(st.get(grp, {}).get(field))
                           for m, (grp, field) in _STAT.items()
                           if st.get(grp, {}).get(field) is not None}
                    if rec:
                        stats[key] = rec
    return stats, events


def grade_from_statsapi(con, date: str, fetch=_get) -> dict:
    """Grade every unresolved prediction for a date and record game ids."""
    pending = store.unresolved(con, date)
    if not pending:
        return {"graded": 0, "pending": 0, "events": 0, "unmatched": []}

    stats, events = final_stats(date, fetch=fetch)
    now = store.utcnow()

    for entity, canon in events.items():
        con.execute(
            "INSERT INTO entity_events "
            "(domain, entity, event_date, canonical_event, resolved_at, resolved_by) "
            "VALUES ('MLB', ?, ?, ?, ?, 'statsapi') "
            "ON CONFLICT(domain, entity, event_date) DO UPDATE SET "
            "canonical_event=excluded.canonical_event, resolved_at=excluded.resolved_at",
            (entity, date, canon, now))
    con.commit()

    graded, unmatched = 0, []
    for p in pending:
        rec = stats.get(p["entity"])
        if rec is None or p["market"] not in rec:
            # Not an error: the game may still be in progress, or the starter
            # was scratched. Left unresolved so the next run retries it.
            unmatched.append(p["entity"])
            continue
        actual = rec[p["market"]]
        side = p["side"] or "over"
        if store.record_result(con, p["id"], actual,
                               result_of(side, p["line"], actual),
                               graded_by="statsapi"):
            graded += 1

    return {"graded": graded, "pending": len(pending), "events": len(events),
            "unmatched": sorted(set(unmatched))}


def apply_source_results(con, results: list[dict], source: str,
                         graded_by: str) -> dict:
    """Apply outcomes a source already computed (Fantasy grades its own).

    Matched on (source, domain, entity, market, event_date, line) — the same
    tuple the prediction id is derived from, so a graded row can only ever land
    on the prediction it actually describes. Notably line is part of the match:
    if the board moved after the prediction was made, the outcome belongs to the
    row priced at that line, not to whichever row is newest.
    """
    applied, unmatched = 0, []
    for r in results:
        if r["line"] is None:
            continue
        row = con.execute(
            "SELECT id, side FROM predictions WHERE source=? AND domain=? AND "
            "entity=? AND market=? AND event_date=? AND line=?",
            (source, r["domain"], r["entity"], r["market"],
             r["event_date"], r["line"])).fetchone()
        if row is None:
            unmatched.append(f"{r['entity']} {r['market']} {r['event_date']}")
            continue
        if store.record_result(con, row["id"], r.get("actual"), r["result"],
                               graded_by=graded_by):
            applied += 1
    return {"applied": applied, "unmatched": sorted(set(unmatched))}
