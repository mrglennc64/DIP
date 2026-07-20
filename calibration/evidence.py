"""Module 7 — Confidence Calibration, plus the evidence gate built on it.

The document's display, verbatim: instead of showing "63%", show

    Historical Accuracy   62%
    Model Confidence      63%
    Calibration Error     1.1%
    Reliability           Excellent

"This tells users how trustworthy the confidence estimate has been
historically." That is the module's whole job: measure stated-vs-realized and
say plainly what the history supports — including "nothing yet", which is a
legitimate and common answer. A calibration module that always produces a
reliability label is worse than none, because it launders absence of evidence
into a word like "Excellent".

Everything here feeds three consumers: the Module 2 calibration component, the
Module 9 "Calibration > 95%" clause, and the Module 10 "calibration has
drifted" alert.
"""
from __future__ import annotations

import math

# Below this many graded outcomes a hit rate is not a measurement — at n=30 the
# 95% interval on a coin flip spans roughly +/-18 points.
MIN_N_ANY = 30
MIN_N_BUCKET = 25

BUCKETS = ((0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.01))

# Reliability label from |calibration error|, per the document's display.
# Thresholds in probability points.
_RELIABILITY = ((0.02, "Excellent"), (0.05, "Good"), (0.10, "Fair"))


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — behaves at small n near 0.5, which is exactly
    where these samples live."""
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def brier(rows: list[dict]) -> float | None:
    r = [x for x in rows if x.get("p") is not None]
    return sum((x["p"] - x["hit"]) ** 2 for x in r) / len(r) if r else None


def log_loss(rows: list[dict]) -> float | None:
    r = [x for x in rows if x.get("p") is not None]
    if not r:
        return None
    tot = 0.0
    for x in r:
        p = min(max(x["p"], 1e-12), 1 - 1e-12)
        tot += -(math.log(p) if x["hit"] else math.log(1 - p))
    return tot / len(r)


def calibration_curve(rows: list[dict]) -> list[dict]:
    """Stated vs realized per confidence bucket — the single most diagnostic
    view. A model can sit at coin-flip Brier overall while its high-confidence
    bucket runs at 40%; only the curve shows the inversion."""
    out = []
    for lo, hi in BUCKETS:
        b = [x for x in rows if x.get("p") is not None and lo <= x["p"] < hi]
        if not b:
            continue
        n, hits = len(b), sum(x["hit"] for x in b)
        stated = sum(x["p"] for x in b) / n
        realized = hits / n
        lo_ci, hi_ci = wilson(hits, n)
        out.append({"bucket": f"{lo:.2f}-{hi:.2f}", "n": n,
                    "stated": stated, "realized": realized,
                    "gap_pts": (realized - stated) * 100,
                    "ci95": (lo_ci, hi_ci),
                    "significant": n >= MIN_N_BUCKET and hi_ci < stated})
    return out


def ranking_skill(curve: list[dict]) -> dict:
    """Does higher stated confidence mean higher realized accuracy? Checked
    separately from calibration because the failures differ: miscalibrated-but-
    ordered is repairable by shifting probabilities; inverted means the ordering
    itself carries no information."""
    usable = [b for b in curve if b["n"] >= MIN_N_BUCKET]
    if len(usable) < 2:
        return {"status": "unknown", "detail":
                f"need 2+ buckets with n>={MIN_N_BUCKET}; have {len(usable)}"}
    lo, hi = usable[0], usable[-1]
    delta = hi["realized"] - lo["realized"]
    if delta < -0.02:
        return {"status": "inverted", "delta": delta, "detail":
                f"highest-confidence bucket realizes {hi['realized']:.1%} vs "
                f"{lo['realized']:.1%} for the lowest — the ordering points "
                f"the wrong way"}
    if abs(delta) <= 0.02:
        return {"status": "flat", "delta": delta, "detail":
                "confidence does not separate outcomes"}
    return {"status": "ordered", "delta": delta, "detail":
            f"realized accuracy rises {delta:+.1%} from lowest to highest bucket"}


def display(model_confidence: float, rows: list[dict]) -> dict:
    """The document's exact four-field display for one confidence value.

    Historical accuracy is bucket-local (graded outcomes within +/-5 points of
    this confidence), because that is the claim being checked — "when the model
    says ~63%, what actually happens?"
    """
    band = [r for r in rows
            if r.get("p") is not None and abs(r["p"] - model_confidence) <= 0.05]
    if len(band) < 10:
        return {"model_confidence": model_confidence,
                "historical_accuracy": None, "calibration_error": None,
                "reliability": "Unknown",
                "note": f"only {len(band)} graded outcomes near "
                        f"{model_confidence:.0%} — not enough to judge"}
    realized = sum(r["hit"] for r in band) / len(band)
    err = abs(realized - model_confidence)
    label = "Poor"
    for cut, name in _RELIABILITY:
        if err <= cut:
            label = name
            break
    return {"model_confidence": model_confidence,
            "historical_accuracy": round(realized, 3),
            "calibration_error": round(err, 3),
            "reliability": label, "n": len(band)}


def assess(rows: list[dict], target_n: int = 200) -> dict:
    """The verdict on a body of graded history — what every downstream module
    gates on."""
    graded = [x for x in rows if x.get("hit") is not None]
    n = len(graded)
    if n == 0:
        return {"status": "no_history", "n": 0, "decided": False,
                "signal": "unknown", "curve": [], "ranking":
                {"status": "unknown", "detail": "no graded history"},
                "headline": "No graded outcomes supplied — quality cannot be "
                            "measured, only structure can."}

    hits = sum(x["hit"] for x in graded)
    realized = hits / n
    lo, hi = wilson(hits, n)
    scored = [x for x in graded if x.get("p") is not None]
    stated = sum(x["p"] for x in scored) / len(scored) if scored else None
    b, ll = brier(graded), log_loss(graded)
    curve = calibration_curve(graded)
    rank = ranking_skill(curve)
    beats_brier = b is not None and b < 0.25 - 0.005

    decided = n >= MIN_N_ANY and (lo > 0.5 or hi < 0.5)
    if n < MIN_N_ANY:
        signal, status = "inconclusive", "gathering"
    elif lo > 0.5 and rank["status"] == "ordered":
        signal, status = "positive", "decided"
    elif hi < 0.5:
        signal, status = "negative", "decided"
    elif rank["status"] == "inverted":
        signal, status = "inverted", "decided"
    else:
        signal, status = "inconclusive", "gathering"

    ci = f"[{lo:.1%}, {hi:.1%}]"
    if signal == "inconclusive":
        need = max(0, target_n - n)
        headline = (f"{n} graded - realized {realized:.1%}, 95% CI {ci} spans "
                    f"50% - NOT DECIDED"
                    + (f" ({need} more to a call)" if need else ""))
    elif signal == "inverted":
        headline = (f"{n} graded - {rank['detail']} - the confidence ranking "
                    f"is actively misleading")
    elif signal == "negative":
        headline = f"{n} graded - realized {realized:.1%}, CI {ci} below 50%"
    else:
        headline = f"{n} graded - realized {realized:.1%}, CI {ci} clears 50%"

    return {"status": status, "decided": decided, "signal": signal,
            "n": n, "target_n": target_n, "progress": min(1.0, n / target_n),
            "realized": realized, "stated": stated,
            "gap_pts": (realized - stated) * 100 if stated is not None else None,
            "ci95": (lo, hi), "brier": b, "log_loss": ll,
            "beats_coinflip_brier": beats_brier,
            "curve": curve, "ranking": rank, "headline": headline}
