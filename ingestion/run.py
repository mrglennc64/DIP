"""Ingest orchestration — what the hourly cron calls.

    python -m ingestion.run ingest --date 2026-07-20
    python -m ingestion.run grade  --date 2026-07-19
    python -m ingestion.run status

Every source is ingested independently and a failure in one is recorded, not
raised. DIP's value comes from seeing sources side by side; an outage at one
upstream must degrade that comparison, never suppress the other source's data
as well. Each attempt lands in ingest_runs whatever happens, because "the feed
returned nothing" and "we never asked" have to be distinguishable afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from . import resolve, store
from .adapters import fantasy, mlbedge

DEFAULT_DB = os.environ.get(
    "DIP_DB", os.path.join(os.path.dirname(__file__), "..", "data", "dip.sqlite3"))

SOURCES = {
    mlbedge.SOURCE: ("stike/mlb-edge", "http_api", f"{mlbedge.BASE}/v2/slate"),
    fantasy.SOURCE: ("Fantasy pick6", "http_json", fantasy.URL),
}


def _ingest_one(con, source: str, date: str, fetcher, mapper) -> dict:
    run_id = store.start_run(con, source, date)
    try:
        payload = fetcher()
        preds, rejected = mapper(payload)
        observed_at = store.utcnow()
        new, seen = store.upsert_predictions(con, preds, observed_at)
        status = "ok" if preds else "empty"
        stamp = (mlbedge if source == mlbedge.SOURCE else fantasy).upstream_stamp(payload)
        store.finish_run(con, run_id, status, rows_seen=seen, rows_new=new,
                         upstream_stamp=stamp)
        for r in rejected:
            # Logged individually and loudly. Silent drops are how an ingest
            # ends up 20% short without anyone being able to say when it started.
            print(f"  [{source}] rejected: {r['why']}", file=sys.stderr)
        return {"source": source, "status": status, "seen": seen, "new": new,
                "rejected": len(rejected), "payload": payload}
    except Exception as e:
        store.finish_run(con, run_id, "error", error=f"{type(e).__name__}: {e}")
        print(f"  [{source}] ERROR {type(e).__name__}: {e}", file=sys.stderr)
        if os.environ.get("DIP_TRACEBACK"):
            traceback.print_exc()
        return {"source": source, "status": "error", "seen": 0, "new": 0,
                "rejected": 0, "payload": None}


def ingest(con, date: str) -> dict:
    out = {}
    out[mlbedge.SOURCE] = _ingest_one(
        con, mlbedge.SOURCE, date,
        lambda: mlbedge.fetch(date),
        lambda p: mlbedge.to_predictions(p, date))

    out[fantasy.SOURCE] = _ingest_one(
        con, fantasy.SOURCE, date,
        fantasy.fetch,
        fantasy.to_predictions)

    # Fantasy grades its own rows; take them rather than re-deriving outcomes
    # the source already settled. DIP only reaches for StatsAPI where nobody
    # upstream graded anything, which is mlb-edge.
    fp = out[fantasy.SOURCE]["payload"]
    if fp:
        res = resolve.apply_source_results(
            con, fantasy.to_results(fp), fantasy.SOURCE, graded_by="fantasy_export")
        out[fantasy.SOURCE]["results_applied"] = res["applied"]

    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ingestion.run")
    ap.add_argument("command", choices=["ingest", "grade", "status", "init"])
    ap.add_argument("--date")
    ap.add_argument("--db", default=DEFAULT_DB)
    a = ap.parse_args(argv)

    con = store.connect(a.db)
    store.init(con)
    for s, (name, kind, ep) in SOURCES.items():
        store.register_source(con, s, name, kind, ep)

    if a.command == "init":
        print(f"initialized {a.db}")
        return 0

    if a.command == "ingest":
        if not a.date:
            ap.error("ingest requires --date")
        res = ingest(con, a.date)
        for s, r in res.items():
            print(f"{s:10s} {r['status']:6s} seen={r['seen']:3d} new={r['new']:3d} "
                  f"rejected={r['rejected']:3d}"
                  + (f" results={r['results_applied']}" if "results_applied" in r else ""))
        shared = store.cross_source(con, a.date)
        print(f"cross-source overlap: {len(shared)} entities seen by >1 source")
        return 0 if any(r["status"] == "ok" for r in res.values()) else 1

    if a.command == "grade":
        if not a.date:
            ap.error("grade requires --date")
        r = resolve.grade_from_statsapi(con, a.date)
        print(f"graded {r['graded']}/{r['pending']} pending, "
              f"{r['events']} game ids resolved")
        if r["unmatched"]:
            print(f"still unresolved ({len(r['unmatched'])}): "
                  f"{', '.join(r['unmatched'][:8])}")
        return 0

    if a.command == "status":
        print(json.dumps({"stats": store.stats(con),
                          "sources": store.source_health(con)}, indent=2))
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
