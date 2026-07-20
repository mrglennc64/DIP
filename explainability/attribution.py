"""Module 6 — Explainability Engine. "Every prediction should explain itself."

The document's example:

    Prediction Score  91
    Why?
      +14%  Elite matchup
      +11%  Strong recent form
      + 7%  Weather
      − 5%  Bullpen strength
      − 3%  Park factor

Two layers of explanation, because DIP has two kinds of information:

1. SCORE ATTRIBUTION (always available): each Module 2 component's signed
   contribution relative to a neutral 50-baseline, in points of the final
   score. Exact by construction — contributions sum to (score - 50·coverage
   adjustment) — so the panel is arithmetic, not narrative.

2. SOURCE REASONS (when the producer ships them): mlb-edge emits reasons[]
   ("elite matchup", "low sample"), and the file adapter preserves any raw
   fields. Passed through verbatim and labelled as the source's own claims —
   DIP does not launder an upstream's stated reasons into measured ones.
"""
from __future__ import annotations

# Component key -> panel label.
_LABELS = {
    "calibration": "Historical calibration at this confidence",
    "historical_accuracy": "Producer's realized accuracy on this market",
    "variance": "Prediction variance (low is good)",
    "feature_completeness": "Feature completeness",
    "model_confidence": "Model confidence",
    "stability": "Prediction stability across readings",
    "recent_performance": "Recent model performance (Brier vs coin flip)",
}


def explain(quality: dict, pred=None) -> dict:
    """The Module 6 panel for one scored prediction."""
    weights, comps = quality["weights"], quality["components"]
    avail = {k: v for k, v in comps.items() if v is not None}
    total_w = sum(weights[k] for k in avail) or 1.0

    # Signed contribution of each component vs a 0.5 neutral value, in points
    # of the 0-100 score, renormalized the same way the score itself is.
    rows = []
    for k, v in avail.items():
        pts = (v - 0.5) * (weights[k] / total_w) * 100
        rows.append({"component": k, "label": _LABELS.get(k, k),
                     "value": round(v, 3), "weight": weights[k],
                     "points": round(pts, 1)})
    rows.sort(key=lambda r: -abs(r["points"]))

    out = {
        "score": quality["score"],
        "version": quality["version"],
        "coverage": quality["coverage"],
        "contributions": rows,
        "unmeasured": [{"component": k, "label": _LABELS.get(k, k),
                        "note": "not computable from available data — shown as "
                                "absent, never defaulted"}
                       for k in quality["missing"]],
    }

    # The producer's own stated reasons, clearly attributed.
    if pred is not None and isinstance(pred.raw, dict):
        reasons = pred.raw.get("reasons")
        if reasons:
            out["source_reasons"] = {"claimed_by": pred.source,
                                     "reasons": list(reasons)}
    return out


def render(explanation: dict) -> str:
    """The document's panel shape, as text."""
    o = [f"Prediction Score  {explanation['score']}",
         "Why?"]
    for r in explanation["contributions"]:
        o.append(f"  {r['points']:+5.1f}  {r['label']}")
    for u in explanation["unmeasured"]:
        o.append(f"      ?  {u['label']} (unmeasured)")
    if "source_reasons" in explanation:
        sr = explanation["source_reasons"]
        o.append(f"  Source ({sr['claimed_by']}) adds: "
                 + "; ".join(sr["reasons"]))
    return "\n".join(o)
