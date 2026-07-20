"""Adapter: stike/mlb-edge — pitcher-strikeout pricing engine.

Upstream: https://strike.perfecthold.online/api  (GET /v2/slate?date=)
Row shape: app/pipeline.py:EdgeRow

DIP is a read-only consumer of this service. That is a hard constraint, not a
convention — /v2/slate and /slate accept `log=true`, which APPENDS to mlb-edge's
own predictions log. A poller that set it would be writing rows into another
system's accuracy record every hour and quietly corrupting the very history DIP
grades it against. The parameter is pinned off below and asserted in tests.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

from ..contract import CONF_CATEGORICAL, Prediction, norm_entity

SOURCE = "mlbedge"
BASE = "https://strike.perfecthold.online/api"
DOMAIN = "MLB"
MARKET = "strikeouts"

# mlb-edge reports confidence as a label, not a number. Mapped to 0-1 midpoints
# and tagged CONF_CATEGORICAL so nothing downstream mistakes it for a
# distribution-derived uncertainty — it is a bucketed verdict from the insight
# layer (app/model/insight.py), and PQI must not pool it with Fantasy's
# calibrated probability as though the two were the same quantity.
_CONFIDENCE = {"high": 0.9, "medium": 0.6, "low": 0.3}

# Which context fields a complete row carries. Completeness is a real PQI input,
# so it is measured against a fixed expectation rather than however many keys
# happened to arrive — otherwise a degraded upstream that drops half its context
# scores as "complete" simply because it also stopped promising the rest.
_CONTEXT = ("k_per_9", "innings_per_start", "opp_k_rate", "park",
            "expected_ks", "bookmaker")


def fetch(date: str, base: str = BASE, timeout: float = 60.0,
          **params) -> dict:
    """GET /v2/slate for a date. `log` is forced off and cannot be overridden."""
    q = {"date": date, **params}
    # Belt and braces: even if a caller passes log=true, it never reaches the
    # wire. The default is already false upstream; this makes it impossible.
    q["log"] = "false"
    url = f"{base}/v2/slate?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def _event_key(row: dict, date: str) -> str:
    """Structural game key from what EdgeRow actually carries.

    There is no game id upstream. `opponent` cannot identify a game on its own —
    for the two starters in one game it holds two different values — but the
    venue hosts one game per date, so date+venue groups a game correctly. This
    is a WITHIN-SOURCE key only; the cross-source join happens through
    entity_events once StatsAPI resolves the real gamePk.
    """
    venue = norm_entity(row.get("venue") or "") or "unknown"
    return f"{date}:{venue.replace(' ', '-')}"


def to_predictions(payload: dict, date: str | None = None) -> tuple[list, list]:
    """Map a /v2/slate payload to contract rows. Returns (predictions, rejected).

    Rejections are RETURNED, not swallowed. A row vanishing quietly between an
    upstream and the ledger is the ingestion bug that never announces itself:
    the totals just come out a little low and nobody notices for a month.
    """
    date = date or payload.get("date") or ""
    preds, rejected = [], []

    for row in payload.get("rows", []):
        # "no_prop"/"no_stats" mean the engine could not price the start. They
        # are not predictions and must not enter the ledger as ones — but they
        # are worth surfacing, because a slate that is suddenly 80% no_stats is
        # a broken upstream wearing an ordinary-looking response.
        if row.get("status") != "ok":
            rejected.append({"row": row, "why": f"status={row.get('status')}"})
            continue
        if row.get("line") is None or row.get("model_prob") is None:
            rejected.append({"row": row, "why": "no line/model_prob"})
            continue

        side = (row.get("side") or "").lower()
        mp = float(row["model_prob"])
        # model_prob is the probability of the side the engine leaned, not of
        # "over". Assigning it to prob_over unconditionally would invert every
        # under-leaning row — a sign error that produces plausible numbers and
        # would only surface as unexplained miscalibration months later.
        if side == "over":
            prob_over, prob_under = mp, 1.0 - mp
        elif side == "under":
            prob_over, prob_under = 1.0 - mp, mp
        else:
            rejected.append({"row": row, "why": f"unrecognized side={side!r}"})
            continue

        present = sum(1 for k in _CONTEXT if row.get(k) is not None)
        name = row.get("pitcher") or ""

        p = Prediction(
            source=SOURCE,
            # No model version is exposed. Recording the empty string is honest;
            # inventing one would let PQI attribute a change to a model build
            # that never existed.
            source_version="",
            domain=DOMAIN,
            entity=norm_entity(name),
            entity_display=name,
            event_key=_event_key(row, date),
            market=MARKET,
            event_date=row.get("date") or date,
            line=float(row["line"]),
            predicted_value=row.get("expected_ks"),
            prob_over=prob_over,
            prob_under=prob_under,
            # The engine is Poisson (its own README flags Ks as mildly
            # UNDER-dispersed relative to Poisson). None means "assume Poisson",
            # which is the truth here rather than a missing value.
            dispersion=None,
            side=side,
            confidence=_CONFIDENCE.get((row.get("confidence") or "").lower()),
            confidence_kind=CONF_CATEGORICAL,
            feature_completeness=present / len(_CONTEXT),
            raw=row,
        )
        problems = p.validate()
        if problems:
            rejected.append({"row": row, "why": "; ".join(problems)})
            continue
        preds.append(p)

    return preds, rejected


def upstream_stamp(payload: dict) -> str:
    """Freshness marker for ingest_runs — how the stale-source alert notices an
    upstream that keeps answering 200 while its content stops moving."""
    return (f"date={payload.get('date')} count={payload.get('count')} "
            f"evaluated={payload.get('evaluated')} bets={payload.get('bets')}")
