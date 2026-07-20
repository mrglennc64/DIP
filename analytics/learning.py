"""Module 8 — Learning Database. "Every prediction becomes a permanent record."

The storage itself lives in db/schema.sql + ingestion/store.py (predictions,
results with residuals, prediction_observations, model_versions — the
document's Store list, with domain-specific context like weather/park/lineup
riding in the frozen `raw` JSON rather than as fixed columns, so the record
works for any domain). This module is the ANALYTICS over that record: the
queries that turn the permanent record into the document's promised
"competitive advantage over time".

Also the bridge that feeds Modules 2 and 7: `history_rows()` converts the
ledger's results into the graded-row shape the calibration and quality modules
consume, so DB-backed and file-backed analysis run through identical code.
"""
from __future__ import annotations

import json


def history_rows(con, domain: str | None = None, market: str | None = None,
                 source: str | None = None) -> list[dict]:
    """Graded rows in the shape calibration/ and scoring/ consume.

    Pushes ('X') are excluded here — there was no side of the line to be on —
    which keeps every rate downstream unbiased, exactly as grade.py does.
    """
    q = """SELECT p.entity, p.market, p.event_date, p.source, p.source_version,
                  p.confidence AS p, p.predicted_value, p.line, p.side,
                  r.actual, r.result
           FROM results r JOIN predictions p ON p.id = r.prediction_id
           WHERE r.result IN ('1','0')"""
    args = []
    for col, val in (("p.domain", domain), ("p.market", market),
                     ("p.source", source)):
        if val:
            q += f" AND {col} = ?"
            args.append(val)
    return [{"entity": r["entity"], "market": r["market"], "date": r["event_date"],
             "source": r["source"], "source_version": r["source_version"],
             "p": r["p"], "predicted": r["predicted_value"], "line": r["line"],
             "side": r["side"], "actual": r["actual"],
             "hit": int(r["result"])}
            for r in con.execute(q, args)]


def residual_summary(con, source: str | None = None) -> dict:
    """Bias and MAE from stored residuals — the point-estimate half of the
    record, complementing the probability half that calibration/ judges."""
    q = """SELECT r.residual FROM results r
           JOIN predictions p ON p.id = r.prediction_id
           WHERE r.residual IS NOT NULL"""
    args = []
    if source:
        q += " AND p.source = ?"
        args.append(source)
    res = [r["residual"] for r in con.execute(q, args)]
    if not res:
        return {"n": 0}
    n = len(res)
    return {"n": n,
            "bias": round(sum(res) / n, 3),                 # actual - predicted
            "mae": round(sum(abs(x) for x in res) / n, 3),
            "rmse": round((sum(x * x for x in res) / n) ** 0.5, 3)}


def by_source(con, domain: str | None = None) -> list[dict]:
    """Per-source scoreboard: n / hit rate / bias / MAE. The cross-source
    comparison no single producer can run on itself."""
    rows = history_rows(con, domain=domain)
    out = {}
    for r in rows:
        s = out.setdefault(r["source"], {"source": r["source"], "n": 0,
                                         "hits": 0, "residuals": []})
        s["n"] += 1
        s["hits"] += r["hit"]
        if r["actual"] is not None and r["predicted"] is not None:
            s["residuals"].append(r["actual"] - r["predicted"])
    result = []
    for s in out.values():
        res = s.pop("residuals")
        s["hit_rate"] = round(s["hits"] / s["n"], 4) if s["n"] else None
        s["bias"] = round(sum(res) / len(res), 3) if res else None
        s["mae"] = round(sum(abs(x) for x in res) / len(res), 3) if res else None
        result.append(s)
    return sorted(result, key=lambda s: -s["n"])


def version_history(con, source: str) -> list[dict]:
    """Model versions seen for a source, with per-version accuracy — the
    document stores Version precisely so a regression can be pinned to the
    build that introduced it."""
    rows = history_rows(con, source=source)
    byv: dict = {}
    for r in rows:
        v = byv.setdefault(r["source_version"] or "(unversioned)",
                           {"version": r["source_version"] or "(unversioned)",
                            "n": 0, "hits": 0})
        v["n"] += 1
        v["hits"] += r["hit"]
    for v in byv.values():
        v["hit_rate"] = round(v["hits"] / v["n"], 4)
    return sorted(byv.values(), key=lambda v: -v["n"])


def raw_context(con, prediction_id: str) -> dict:
    """The frozen upstream row — weather, park, lineup slot, pitch mix, features
    — exactly as the producer shipped it. Never re-derived."""
    r = con.execute("SELECT raw FROM predictions WHERE id = ?",
                    (prediction_id,)).fetchone()
    return json.loads(r["raw"]) if r else {}
