"""The ledger — SQLite behind a small, deliberately narrow API.

SQLite rather than Postgres because the honest volume is ~30 predictions/day per
source: a few tens of thousands of rows a year. WAL mode covers one hourly
writer plus dashboard readers without blocking, and it keeps deployment to a
single file with no server to run, back up, or lose.

The write path enforces one invariant everything else depends on:
a prediction row is FROZEN at first sight and never updated. Later polls append
to prediction_observations. Upstreams re-project during the day — mlb-edge's
/v2/slate recomputes past dates using current-season stats — so an upsert that
overwrote in place would rewrite what we claim to have believed beforehand,
using information that did not exist then. That is not a hypothetical: it is the
documented leak that drove a dispersion fit to fake-Poisson upstream, and an
accuracy ledger that permits it is measuring nothing.
"""
from __future__ import annotations

import datetime
import json
import os
import sqlite3

SCHEMA = os.path.join(os.path.dirname(__file__), "..", "db", "schema.sql")

_PRED_COLS = (
    "id", "source", "source_version", "domain", "entity", "entity_display",
    "event_key", "market", "event_date", "predicted_value", "line",
    "prob_over", "prob_under", "prob_uncalibrated", "variance", "dispersion", "side",
    "confidence", "confidence_kind", "feature_completeness",
    "first_seen_at", "raw",
)


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def connect(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init(con: sqlite3.Connection) -> None:
    """Idempotent — every statement in schema.sql is CREATE ... IF NOT EXISTS."""
    with open(SCHEMA, encoding="utf-8") as f:
        con.executescript(f.read())
    con.commit()


def register_source(con: sqlite3.Connection, source: str, display_name: str,
                    kind: str, endpoint: str = "", notes: str = "") -> None:
    con.execute(
        "INSERT INTO prediction_sources (source, display_name, kind, endpoint, notes) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(source) DO UPDATE SET "
        "display_name=excluded.display_name, kind=excluded.kind, "
        "endpoint=excluded.endpoint, notes=excluded.notes",
        (source, display_name, kind, endpoint, notes))
    con.commit()


def upsert_predictions(con: sqlite3.Connection, preds: list, observed_at: str
                       ) -> tuple[int, int]:
    """Freeze new predictions, record an observation for every one seen.

    Returns (new, seen). Re-running an identical ingest yields new=0 and adds no
    observation rows either — the UNIQUE(prediction_id, observed_at) key makes
    the whole operation idempotent per timestamp, so a retried cron run cannot
    inflate the stability variance PQI reads off this table.
    """
    new = 0
    for p in preds:
        row = p.to_row(first_seen_at=observed_at)
        cur = con.execute(
            f"INSERT OR IGNORE INTO predictions ({','.join(_PRED_COLS)}) "
            f"VALUES ({','.join('?' * len(_PRED_COLS))})",
            tuple(row[c] for c in _PRED_COLS))
        new += cur.rowcount

        con.execute(
            "INSERT OR IGNORE INTO prediction_observations "
            "(prediction_id, observed_at, predicted_value, prob_over, confidence, line) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (p.id, observed_at, p.predicted_value, p.prob_over, p.confidence, p.line))

        if p.source_version:
            con.execute(
                "INSERT INTO model_versions (source, version, first_seen_at, last_seen_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(source, version) DO UPDATE SET "
                "last_seen_at=excluded.last_seen_at",
                (p.source, p.source_version, observed_at, observed_at))
    con.commit()
    return new, len(preds)


def record_result(con: sqlite3.Connection, prediction_id: str, actual: float | None,
                  result: str, graded_by: str, settled_at: str | None = None) -> bool:
    """Attach an outcome. Returns True if this was a new grade.

    Write-once: an existing result is never overwritten. A settled game does not
    unsettle, so a second grade means either a duplicate run (harmless) or an
    upstream contradicting itself (which must be investigated, not silently
    applied on top of the number already published).
    """
    pred = con.execute("SELECT predicted_value FROM predictions WHERE id = ?",
                       (prediction_id,)).fetchone()
    if pred is None:
        return False
    residual = None
    if actual is not None and pred["predicted_value"] is not None:
        residual = actual - pred["predicted_value"]
    cur = con.execute(
        "INSERT OR IGNORE INTO results "
        "(prediction_id, actual, result, residual, settled_at, graded_by) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (prediction_id, actual, result, residual, settled_at or utcnow(), graded_by))
    con.commit()
    return cur.rowcount > 0


def start_run(con: sqlite3.Connection, source: str, event_date: str) -> int:
    cur = con.execute(
        "INSERT INTO ingest_runs (source, event_date, started_at, status) "
        "VALUES (?, ?, ?, 'running')", (source, event_date, utcnow()))
    con.commit()
    return cur.lastrowid


def finish_run(con: sqlite3.Connection, run_id: int, status: str, rows_seen: int = 0,
               rows_new: int = 0, rows_updated: int = 0,
               upstream_stamp: str | None = None, error: str | None = None) -> None:
    con.execute(
        "UPDATE ingest_runs SET finished_at=?, status=?, rows_seen=?, rows_new=?, "
        "rows_updated=?, upstream_stamp=?, error=? WHERE id=?",
        (utcnow(), status, rows_seen, rows_new, rows_updated, upstream_stamp,
         error, run_id))
    con.commit()


def source_health(con: sqlite3.Connection) -> list[dict]:
    """Last run per source — what GET /sources answers and what the stale alert
    watches. A source that stops returning rows is indistinguishable from a
    quiet day if you only look at the predictions table, which is exactly why
    this reads ingest_runs instead."""
    rows = con.execute("""
        SELECT r.* FROM ingest_runs r
        JOIN (SELECT source, MAX(started_at) AS m FROM ingest_runs GROUP BY source) t
          ON r.source = t.source AND r.started_at = t.m
        ORDER BY r.source
    """).fetchall()
    return [dict(r) for r in rows]


def cross_source(con: sqlite3.Connection, event_date: str) -> list[dict]:
    """Predictions grouped by (entity, market) with more than one source.

    This query is the entire point of Phase 1 — the comparison no upstream can
    perform, because none of them can see the others.
    """
    rows = con.execute("""
        SELECT domain, entity, market, event_key,
               COUNT(DISTINCT source) AS n_sources,
               GROUP_CONCAT(source)   AS sources,
               GROUP_CONCAT(id)       AS ids
        FROM predictions
        WHERE event_date = ?
        GROUP BY domain, entity, market
        HAVING n_sources > 1
        ORDER BY entity
    """, (event_date,)).fetchall()
    return [dict(r) for r in rows]


def unresolved(con: sqlite3.Connection, event_date: str) -> list[dict]:
    rows = con.execute("""
        SELECT p.* FROM predictions p
        LEFT JOIN results r ON r.prediction_id = p.id
        WHERE p.event_date = ? AND r.prediction_id IS NULL
    """, (event_date,)).fetchall()
    return [dict(r) for r in rows]


def stats(con: sqlite3.Connection, event_date: str | None = None) -> dict:
    where, args = ("WHERE event_date = ?", (event_date,)) if event_date else ("", ())
    preds = con.execute(f"SELECT COUNT(*) c FROM predictions {where}", args).fetchone()["c"]
    obs = con.execute("SELECT COUNT(*) c FROM prediction_observations").fetchone()["c"]
    res = con.execute("SELECT COUNT(*) c FROM results").fetchone()["c"]
    by_src = {r["source"]: r["c"] for r in con.execute(
        f"SELECT source, COUNT(*) c FROM predictions {where} GROUP BY source", args)}
    return {"predictions": preds, "observations": obs, "results": res,
            "by_source": by_src}
