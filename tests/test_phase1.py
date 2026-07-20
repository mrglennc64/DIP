"""Phase 1 verification — the checks the plan names, plus the traps around them.

Run:  python -m tests.test_phase1     (from decision-platform/)

No network. Both adapters are driven from captured-shape fixtures so the suite
is deterministic and runnable with the upstreams down.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion import resolve, store                      # noqa: E402
from ingestion.adapters import fantasy, mlbedge           # noqa: E402
from ingestion.contract import norm_entity, prediction_id  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{extra}]" if extra else ""))
    if not cond:
        FAILS.append(name)
    return cond


# ----------------------------------------------------------------- fixtures --
# Shapes copied from app/pipeline.py:EdgeRow and the /v2/slate envelope.
SLATE = {
    "date": "2026-07-20", "count": 4, "evaluated": 4, "skipped": 0, "bets": 1,
    "rows": [
        {"date": "2026-07-20", "pitcher": "Martín Pérez", "opponent": "HOU",
         "venue": "Minute Maid Park", "status": "ok", "expected_ks": 5.9,
         "line": 5.5, "bookmaker": "draftkings", "side": "over",
         "model_prob": 0.58, "fair_prob": 0.53, "over_odds": -115,
         "under_odds": -105, "edge": 0.05, "kelly": 0.03, "recommendation": "Lean",
         "confidence": "Medium", "signal": "lean", "reasons": ["elite matchup"],
         "k_per_9": 9.1, "innings_per_start": 5.4, "opp_k_rate": 0.24, "park": 1.02},
        # UNDER lean — model_prob is the probability of the LEANED side, so this
        # must land as prob_over = 1 - 0.61, not 0.61.
        {"date": "2026-07-20", "pitcher": "Chase Burns", "opponent": "MIL",
         "venue": "American Family Field", "status": "ok", "expected_ks": 6.4,
         "line": 7.5, "bookmaker": "draftkings", "side": "under",
         "model_prob": 0.61, "fair_prob": 0.55, "over_odds": 120,
         "under_odds": -140, "edge": 0.06, "kelly": 0.04, "recommendation": "Lean",
         "confidence": "High", "signal": "lean", "reasons": [],
         "k_per_9": 11.2, "innings_per_start": 5.1, "opp_k_rate": 0.26, "park": 0.99},
        {"date": "2026-07-20", "pitcher": "Nobody Priced", "opponent": "SD",
         "venue": "Petco Park", "status": "no_prop"},
        {"date": "2026-07-20", "pitcher": "No Stats Guy", "opponent": "SEA",
         "venue": "T-Mobile Park", "status": "no_stats"},
    ],
}

DIPJSON = {
    "schema": "dip-export/1", "source": "fantasy", "domain": "MLB",
    "date": "2026-07-20", "board_status": "frozen",
    "frozen_at": "Jul 20 15:04 UTC", "graded_through": "2026-07-19",
    "dispersion_r": 16.6, "unmatched": [],
    "predictions": [
        {"entity": "martin perez", "player": "Martín Pérez", "event_key": "TEX@HOU",
         "market": "strikeouts", "platform": "prizepicks", "side": "more",
         "line": 5.5, "predicted": 6.1, "p_more": 0.612, "p_less": 0.388,
         "p": 0.612, "p_more_raw": 0.634, "p_uncal": 0.601, "rw_proj": 6.0,
         "rw_agree": True, "mu_source": "kmodel", "mu_version": "k3",
         "bench_proj": 5.8, "bench_source": "mlbedge_slate"},
        {"entity": "chase burns", "player": "Chase Burns", "event_key": "CIN@MIL",
         "market": "strikeouts", "platform": "dk_pick6", "side": "less",
         "line": 7.5, "predicted": 6.9, "p_more": 0.421, "p_less": 0.579,
         "p": 0.579, "p_more_raw": 0.44, "p_uncal": 0.571, "rw_proj": None,
         "rw_agree": None, "mu_source": "kmodel", "mu_version": "k3",
         "bench_proj": None, "bench_source": ""},
    ],
    "results": [
        {"date": "2026-07-19", "entity": "zack wheeler", "player": "Zack Wheeler",
         "event_key": "PHI@NYM", "market": "strikeouts", "side": "more",
         "line": 6.5, "predicted": 7.1, "actual": 8.0, "result": "1",
         "mu_source": "kmodel"},
        {"date": "2026-07-19", "entity": "tarik skubal", "player": "Tarik Skubal",
         "event_key": "DET@CLE", "market": "strikeouts", "side": "more",
         "line": 7.0, "predicted": 7.4, "actual": 7.0, "result": "X",
         "mu_source": "kmodel"},
    ],
}


# ------------------------------------------------------------------- tests --
print("--- contract ---")
check("normalizer matches Fantasy's byte-for-byte",
      norm_entity("Martín Pérez") == "martin perez", norm_entity("Martín Pérez"))
check("line is part of prediction identity (a moved line is a new prediction)",
      prediction_id("s", "MLB", "x", "k", 5.5, "d")
      != prediction_id("s", "MLB", "x", "k", 6.5, "d"))
check("id stable across calls",
      prediction_id("s", "MLB", "x", "k", 5.5, "d")
      == prediction_id("s", "MLB", "x", "k", 5.5, "d"))

print("\n--- mlbedge adapter ---")
mp, mr = mlbedge.to_predictions(SLATE)
check("prices only status=ok rows", len(mp) == 2, f"{len(mp)} kept")
check("non-ok rows rejected WITH reason, not dropped silently",
      len(mr) == 2 and all(r["why"] for r in mr))
perez = [p for p in mp if p.entity == "martin perez"][0]
burns = [p for p in mp if p.entity == "chase burns"][0]
check("over lean -> prob_over = model_prob", abs(perez.prob_over - 0.58) < 1e-9,
      perez.prob_over)
check("UNDER lean -> prob_over = 1 - model_prob (sign trap)",
      abs(burns.prob_over - 0.39) < 1e-9, burns.prob_over)
check("probs sum to 1", all(abs(p.prob_over + p.prob_under - 1) < 1e-9 for p in mp))
check("Poisson recorded as dispersion=None, not a fake value",
      all(p.dispersion is None for p in mp))
check("categorical confidence tagged as such",
      perez.confidence_kind == "categorical" and perez.confidence == 0.6)
check("same venue+date groups a game", burns.event_key == "2026-07-20:american-family-field",
      burns.event_key)

# The constraint that matters most: never write to an upstream's ledger.
sent = {}


def _fake_urlopen(url, timeout=None):
    sent["url"] = url
    raise RuntimeError("stop before network")


import urllib.request  # noqa: E402
_real = urllib.request.urlopen
urllib.request.urlopen = _fake_urlopen
for attempt in ({}, {"log": "true"}, {"log": True}):
    try:
        mlbedge.fetch("2026-07-20", **attempt)
    except RuntimeError:
        pass
    check(f"log=false forced on the wire (caller passed {attempt or 'nothing'})",
          "log=false" in sent["url"] and "log=true" not in sent["url"], sent["url"])
urllib.request.urlopen = _real

print("\n--- fantasy adapter ---")
fp, fr = fantasy.to_predictions(DIPJSON)
check("2 predictions mapped", len(fp) == 2 and not fr)
fperez = [p for p in fp if p.entity == "martin perez"][0]
check("dispersion carried (NB, not Poisson)", fperez.dispersion == 16.6)
check("uncalibrated probability carried", fperez.prob_uncalibrated == 0.601)
check("probability-kind confidence tagged distinctly",
      fperez.confidence_kind == "probability")
check("source_version carried", fperez.source_version == "k3")
bad = dict(DIPJSON, schema="dip-export/2")
try:
    fantasy.to_predictions(bad)
    check("refuses unsupported payload major", False)
except ValueError:
    check("refuses unsupported payload major", True)

print("\n--- ledger ---")
db = os.path.join(tempfile.mkdtemp(), "dip.sqlite3")
con = store.connect(db)
store.init(con)
store.register_source(con, "mlbedge", "stike/mlb-edge", "http_api")
store.register_source(con, "fantasy", "Fantasy pick6", "http_json")

t1 = "2026-07-20T15:00:00+00:00"
n1, s1 = store.upsert_predictions(con, mp, t1)
n2, s2 = store.upsert_predictions(con, fp, t1)
check("4 predictions land from 2 sources", n1 + n2 == 4, f"{n1}+{n2}")

# THE Phase 1 milestone: the comparison no upstream can make.
shared = store.cross_source(con, "2026-07-20")
check("both sources joinable on the same pitchers", len(shared) == 2,
      [r["entity"] for r in shared])
check("overlap names both sources",
      all(sorted(r["sources"].split(",")) == ["fantasy", "mlbedge"] for r in shared))

# Idempotency — a retried cron run must not inflate anything.
before = store.stats(con)
store.upsert_predictions(con, mp, t1)
store.upsert_predictions(con, fp, t1)
after = store.stats(con)
check("re-ingest at same timestamp changes nothing", before == after,
      f"{before} -> {after}")

# Freeze-on-first-sight: a later poll must not rewrite what was believed.
import dataclasses  # noqa: E402
moved = dataclasses.replace(perez, predicted_value=99.0)
store.upsert_predictions(con, [moved], "2026-07-20T16:00:00+00:00")
row = con.execute("SELECT predicted_value FROM predictions WHERE id=?",
                  (perez.id,)).fetchone()
check("frozen row survives a re-projection (no backwards leak)",
      row["predicted_value"] == 5.9, row["predicted_value"])
obs = con.execute("SELECT COUNT(*) c FROM prediction_observations WHERE prediction_id=?",
                  (perez.id,)).fetchone()["c"]
check("but the new reading IS kept as an observation (PQI stability input)",
      obs == 2, obs)

print("\n--- grading ---")
check("whole line + exact actual = push, not loss", resolve.result_of("more", 7.0, 7.0) == "X")
check("half line can never push", resolve.result_of("more", 6.5, 7) == "1")
check("over/under vocabulary == more/less",
      resolve.result_of("over", 5.5, 6) == resolve.result_of("more", 5.5, 6) == "1")
check("under lean correct when actual below line", resolve.result_of("under", 7.5, 6) == "1")

# Fantasy's own grades applied to Fantasy rows (its results are for 07-19, so
# seed matching predictions first).
past = [dataclasses.replace(fperez, entity="zack wheeler", entity_display="Zack Wheeler",
                            event_date="2026-07-19", line=6.5, predicted_value=7.1,
                            side="more"),
        dataclasses.replace(fperez, entity="tarik skubal", entity_display="Tarik Skubal",
                            event_date="2026-07-19", line=7.0, predicted_value=7.4,
                            side="more")]
store.upsert_predictions(con, past, t1)
ap = resolve.apply_source_results(con, fantasy.to_results(DIPJSON), "fantasy",
                                  graded_by="fantasy_export")
check("source-graded results applied", ap["applied"] == 2, ap)
pushrow = con.execute(
    "SELECT r.result FROM results r JOIN predictions p ON p.id=r.prediction_id "
    "WHERE p.entity='tarik skubal'").fetchone()
check("push survives the round trip as 'X' (never collapsed to a loss)",
      pushrow["result"] == "X", pushrow["result"])
resid = con.execute(
    "SELECT r.residual FROM results r JOIN predictions p ON p.id=r.prediction_id "
    "WHERE p.entity='zack wheeler'").fetchone()["residual"]
check("residual = actual - predicted", abs(resid - (8.0 - 7.1)) < 1e-9, resid)

# Write-once: a settled game does not unsettle.
wid = con.execute("SELECT id FROM predictions WHERE entity='zack wheeler'").fetchone()["id"]
check("existing result never overwritten",
      store.record_result(con, wid, 2.0, "0", "bogus") is False)
check("...and the original stands",
      con.execute("SELECT result FROM results WHERE prediction_id=?",
                  (wid,)).fetchone()["result"] == "1")

print("\n--- statsapi grading (offline fixture) ---")
FAKE = {
    "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=2026-07-20": {
        "dates": [{"games": [
            {"gamePk": 745812, "status": {"abstractGameState": "Final"}},
            {"gamePk": 745999, "status": {"abstractGameState": "Live"}},
        ]}]},
    "https://statsapi.mlb.com/api/v1/game/745812/boxscore": {
        "teams": {"home": {"players": {"ID1": {
            "person": {"fullName": "Martín Pérez"},
            "stats": {"pitching": {"strikeOuts": 7}}}}},
            "away": {"players": {}}}},
}
res = resolve.grade_from_statsapi(con, "2026-07-20", fetch=lambda u, timeout=60.0: FAKE[u])
# BOTH sources' Perez rows grade off the one boxscore — grading is a property of
# the event, not of who predicted it. That is the intent: DIP settles every
# unresolved row it can, and record_result is write-once, so Fantasy's own grade
# arriving later cannot contradict or duplicate this one.
check("every unresolved row for a Final game is graded, across sources",
      res["graded"] == 2, res)
graders = {r["graded_by"] for r in con.execute(
    "SELECT DISTINCT graded_by FROM results")}
check("grader is recorded per result", graders == {"statsapi", "fantasy_export"}, graders)
check("row whose game is not Final stays unresolved for the next run",
      res["unmatched"] == ["chase burns"], res["unmatched"])
canon = con.execute("SELECT canonical_event FROM entity_events WHERE entity='martin perez'"
                    " AND event_date='2026-07-20'").fetchone()["canonical_event"]
check("canonical gamePk resolved for cross-source correlation",
      canon == "mlb:745812", canon)
check("live game not read (partial stats would grade as a confident loss)",
      "745999" not in str(res))

print("\n--- source health ---")
rid = store.start_run(con, "mlbedge", "2026-07-20")
store.finish_run(con, rid, "ok", rows_seen=2, rows_new=2, upstream_stamp="count=4")
h = store.source_health(con)
check("last run per source reported", len(h) >= 1 and h[0]["status"] in ("ok", "running"))

con.close()
print("\n" + (f"{len(FAILS)} FAILURES: " + ", ".join(FAILS) if FAILS else "ALL PASS"))
sys.exit(1 if FAILS else 0)
