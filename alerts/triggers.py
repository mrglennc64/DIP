"""Module 10 — Alerts. "Instead of generic notifications, trigger meaningful ones."

The document's five examples, each implemented as a named trigger:

    1. Today's highest-quality prediction.
    2. Calibration has drifted.
    3. Model accuracy improved this week.
    4. Weather materially changed a projection.   (generalized: any material
       projection change — weather is one cause the source may name)
    5. Bullpen announcement changed confidence.   (generalized: any material
       confidence change)

4 and 5 are generalized because DIP consumes predictions rather than making
them: it cannot know WHY a projection moved, only that it moved and by how
much. The prediction_observations table exists precisely for this — it holds
every reading of the same prediction across polls, so a material move is
detectable and the source's stated reason (when it ships one) rides along.

Every alert carries a dedupe_key so one CONDITION fires one alert per day, not
one per poll. Ingest runs hourly; without the key, a drifted calibration would
page 24 times for one fact.
"""
from __future__ import annotations

import json

# Material-change thresholds, in the prediction's own units.
PROJECTION_MOVE_MIN = 0.075     # 7.5% relative move in predicted_value
CONFIDENCE_MOVE_MIN = 0.05      # 5 points of probability
CALIBRATION_DRIFT_MIN = 0.05    # |stated - realized| gap, probability points
ACCURACY_IMPROVE_MIN = 0.03     # week-over-week hit-rate gain


def _alert(kind, severity, subject, body, dedupe_key):
    return {"kind": kind, "severity": severity, "subject": subject,
            "body": body, "dedupe_key": dedupe_key}


def highest_quality_today(scored: list[dict], date: str) -> list[dict]:
    """1. Today's highest-quality prediction."""
    usable = [s for s in scored if s["quality"]["score"] is not None]
    if not usable:
        return []
    best = max(usable, key=lambda s: s["quality"]["score"])
    p, q = best["pred"], best["quality"]
    return [_alert(
        "highest_quality", "info",
        f"Highest-quality prediction today: {p.entity_display}",
        f"{p.entity_display} {p.market} {p.side or ''} {p.line:g} — "
        f"score {q['score']} ({q['coverage']:.0%} coverage, {q['version']})",
        f"highest_quality:{date}")]


def calibration_drift(assessment: dict, date: str) -> list[dict]:
    """2. Calibration has drifted."""
    out = []
    gap = assessment.get("gap_pts")
    if gap is not None and abs(gap) / 100 >= CALIBRATION_DRIFT_MIN:
        out.append(_alert(
            "calibration_drift", "warning",
            f"Calibration drift: stated vs realized gap {gap:+.1f} pts",
            assessment["headline"],
            f"calibration_drift:{date}"))
    if assessment.get("ranking", {}).get("status") == "inverted":
        out.append(_alert(
            "ranking_inverted", "critical",
            "Confidence ranking is INVERTED",
            assessment["ranking"]["detail"],
            f"ranking_inverted:{date}"))
    return out


def accuracy_improved(this_week: dict, last_week: dict, date: str) -> list[dict]:
    """3. Model accuracy improved this week. Both weeks must individually be
    measurements (n >= 30) — comparing two noise estimates produces a weekly
    coin-flip announcement in each direction."""
    a, b = this_week, last_week
    if (a.get("n", 0) < 30 or b.get("n", 0) < 30
            or a.get("realized") is None or b.get("realized") is None):
        return []
    gain = a["realized"] - b["realized"]
    if gain >= ACCURACY_IMPROVE_MIN:
        return [_alert(
            "accuracy_improved", "info",
            f"Model accuracy improved: {b['realized']:.1%} -> {a['realized']:.1%}",
            f"+{gain:.1%} on n={a['n']} this week vs n={b['n']} last week",
            f"accuracy_improved:{date}")]
    return []


def material_changes(con, date: str) -> list[dict]:
    """4 + 5. A projection or confidence that moved materially between polls.

    Reads prediction_observations: first vs latest reading per prediction for
    the date. The source's own stated reason (reasons[] etc.) is attached from
    the frozen raw row when present — DIP reports the move as fact and the
    cause as the source's claim.
    """
    out = []
    rows = con.execute("""
        SELECT p.id, p.entity_display, p.market, p.line, p.raw,
               MIN(o.observed_at) AS first_at, MAX(o.observed_at) AS last_at
        FROM predictions p JOIN prediction_observations o
             ON o.prediction_id = p.id
        WHERE p.event_date = ?
        GROUP BY p.id HAVING COUNT(*) > 1
    """, (date,)).fetchall()

    for r in rows:
        first = con.execute(
            "SELECT predicted_value, confidence FROM prediction_observations "
            "WHERE prediction_id=? AND observed_at=?", (r["id"], r["first_at"])).fetchone()
        last = con.execute(
            "SELECT predicted_value, confidence FROM prediction_observations "
            "WHERE prediction_id=? AND observed_at=?", (r["id"], r["last_at"])).fetchone()

        reasons = ""
        try:
            raw = json.loads(r["raw"])
            if raw.get("reasons"):
                reasons = f" Source cites: {'; '.join(raw['reasons'])}."
        except Exception:
            pass

        pv0, pv1 = first["predicted_value"], last["predicted_value"]
        if pv0 and pv1 and abs(pv1 - pv0) / abs(pv0) >= PROJECTION_MOVE_MIN:
            out.append(_alert(
                "projection_changed", "warning",
                f"Projection moved: {r['entity_display']} {r['market']}",
                f"{pv0:g} -> {pv1:g} ({(pv1 - pv0) / pv0:+.1%}) between polls."
                + reasons,
                f"projection_changed:{r['id']}:{date}"))

        c0, c1 = first["confidence"], last["confidence"]
        if c0 is not None and c1 is not None and abs(c1 - c0) >= CONFIDENCE_MOVE_MIN:
            out.append(_alert(
                "confidence_changed", "warning",
                f"Confidence moved: {r['entity_display']} {r['market']}",
                f"{c0:.1%} -> {c1:.1%} ({c1 - c0:+.1%}) between polls." + reasons,
                f"confidence_changed:{r['id']}:{date}"))
    return out


def store(con, alerts: list[dict]) -> int:
    """Persist, deduped. Returns how many were actually new."""
    new = 0
    for a in alerts:
        cur = con.execute(
            "INSERT OR IGNORE INTO alerts "
            "(created_at, kind, severity, subject, body, dedupe_key) "
            "VALUES (datetime('now'), ?, ?, ?, ?, ?)",
            (a["kind"], a["severity"], a["subject"], a["body"], a["dedupe_key"]))
        new += cur.rowcount
    con.commit()
    return new
