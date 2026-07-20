"""Module 2 — Quality Score. Every prediction gets its own score.

The document's weighting, verbatim:

    Calibration                 30%
    Historical Accuracy         30%
    Variance                    15%
    Feature Completeness        15%
    Recent Model Performance    10%

Page 11 additionally offers a PQI variant (0.30 calibration / 0.25 confidence /
0.20 completeness / 0.15 stability / 0.10 recent). Both are implemented and
versioned; Module 2's is the default because it is the module definition, and
the PQI is the named alternative. The document is explicit that neither set is
validated — "weights should be learned or tuned using historical data and
evaluated out-of-sample" — so every component is stored per prediction and each
score records which version produced it. A refit adds a version; it never
rewrites a score already shown.

Two honesty rules carried throughout:
1. A component that cannot be computed is None, never a default. Substituting
   0.5 for "no history" makes an unmeasured prediction look like an average one.
2. Every score reports COVERAGE — the share of total weight actually computable.
   A 92/100 built from 45% of the weight is a different object from a 92 built
   from all of it.
"""
from __future__ import annotations

import statistics

DEFAULT_VERSION = "m2_v1"

WEIGHTS = {
    # Module 2, page 3.
    "m2_v1": {
        "calibration": 0.30,
        "historical_accuracy": 0.30,
        "variance": 0.15,
        "feature_completeness": 0.15,
        "recent_performance": 0.10,
    },
    # Page 11 PQI example.
    "pqi_v1": {
        "calibration": 0.30,
        "model_confidence": 0.25,
        "feature_completeness": 0.20,
        "stability": 0.15,
        "recent_performance": 0.10,
    },
}


# ------------------------------------------------------------- components ---
def calibration_component(pred, history_index) -> float | None:
    """How well this source/market has done at THIS confidence level.

    Bucket-local: a model can be calibrated on average yet badly overconfident
    in exactly the band its recommendations live in. 1.0 when realized matches
    stated; decays in either direction.
    """
    p = pred.confidence
    if p is None or not history_index:
        return None
    rows = history_index.get((pred.market,)) or history_index.get(("*",)) or []
    band = [r for r in rows if r.get("p") is not None and abs(r["p"] - p) <= 0.05]
    if len(band) < 10:
        return None
    realized = sum(r["hit"] for r in band) / len(band)
    stated = sum(r["p"] for r in band) / len(band)
    return max(0.0, 1.0 - abs(realized - stated) / 0.25)


def historical_accuracy_component(pred, history_index) -> float | None:
    """The producer's realized hit rate on this market, scaled.

    50% (coin flip) -> 0; 65% or better -> 1. Distinct from calibration: a model
    can be honest about a mediocre rate (calibrated, low accuracy) or lucky and
    overconfident (accurate lately, miscalibrated). Module 2 weights them
    equally and separately, at 30% each.
    """
    if not history_index:
        return None
    rows = history_index.get((pred.market,)) or history_index.get(("*",)) or []
    if len(rows) < 30:            # below this, a rate is noise, not a measure
        return None
    rate = sum(r["hit"] for r in rows) / len(rows)
    return min(1.0, max(0.0, (rate - 0.50) / 0.15))


def variance_component(pred) -> float | None:
    """Module 1's variance field, inverted: low variance scores high.

    Scaled against the prediction's own magnitude (coefficient of variation) so
    a variance of 4 on a 6-strikeout projection and on a 90k-revenue forecast
    are judged on the same footing. Falls back to the NB dispersion when a
    source ships that instead: Var = mu(1 + mu/r).
    """
    var = pred.variance
    if var is None and pred.dispersion and pred.predicted_value:
        mu = pred.predicted_value
        var = mu * (1 + mu / pred.dispersion)
    if var is None or not pred.predicted_value:
        return None
    cv = (var ** 0.5) / abs(pred.predicted_value)
    return max(0.0, 1.0 - cv)     # cv >= 1 (sd as large as the value) -> 0


def completeness_component(pred) -> float | None:
    return pred.feature_completeness


def confidence_component(pred) -> float | None:
    """PQI variant only: distance from even money, normalized. A restatement of
    the prediction, not evidence about it — which is why Module 2 replaced it
    with historical accuracy."""
    if pred.confidence is None:
        return None
    return min(1.0, max(0.0, (pred.confidence - 0.5) * 2))


def stability_component(pred, observations) -> float | None:
    """PQI variant: movement of the same prediction across repeated readings.
    None with fewer than two observations — an unrepeated measurement is not a
    steady one."""
    obs = (observations or {}).get(pred.id)
    if not obs or len(obs) < 2:
        return None
    vals = [o for o in obs if o is not None]
    if len(vals) < 2:
        return None
    mean = statistics.fmean(vals)
    if mean == 0:
        return None
    return max(0.0, 1.0 - (statistics.pstdev(vals) / abs(mean)) * 5)


def recent_performance_component(assessment) -> float | None:
    """Rolling Brier margin over the 0.25 coin-flip baseline; 0.02 of real
    improvement reaches the top of the range."""
    if not assessment or assessment.get("brier") is None:
        return None
    return min(1.0, max(0.0, (0.25 - assessment["brier"]) / 0.02))


# ------------------------------------------------------------------ score ---
def score(pred, history_index=None, observations=None, assessment=None,
          version: str = DEFAULT_VERSION) -> dict:
    weights = WEIGHTS[version]
    all_components = {
        "calibration": calibration_component(pred, history_index),
        "historical_accuracy": historical_accuracy_component(pred, history_index),
        "variance": variance_component(pred),
        "feature_completeness": completeness_component(pred),
        "model_confidence": confidence_component(pred),
        "stability": stability_component(pred, observations),
        "recent_performance": recent_performance_component(assessment),
    }
    comps = {k: all_components[k] for k in weights}
    avail = {k: v for k, v in comps.items() if v is not None}
    total_w = sum(weights[k] for k in avail)
    coverage = total_w / sum(weights.values())
    value = (sum(weights[k] * v for k, v in avail.items()) / total_w * 100
             if total_w else None)

    return {
        "version": version,
        "score": round(value, 1) if value is not None else None,
        "coverage": round(coverage, 3),
        "components": comps,
        "weights": dict(weights),
        "missing": sorted(k for k, v in comps.items() if v is None),
    }


def index_history(rows: list[dict]) -> dict:
    """Group graded rows for per-market lookup, plus an all-markets fallback."""
    idx: dict = {("*",): list(rows)}
    for r in rows:
        idx.setdefault((r.get("market", "value"),), []).append(r)
    return idx
