"""Module 9 — Recommendation Engine. "Separate prediction from recommendation."

The document's example rule, verbatim:

    if QualityScore > 90
    and Calibration > 95%
    and Variance Low
    and Correlation Low
    then Recommendation = High

Implemented at the document's thresholds, as versioned data. "This allows
recommendations to evolve without changing the prediction model" — so the rules
live in a table of clauses, every evaluation records which version ran and
which clauses passed or failed, and changing policy means adding a version,
never editing history.

The two qualitative clauses need definitions to be executable, recorded here so
they are inspectable rather than implicit:
  "Variance Low"    -> variance component >= 0.7 (cv <= 0.3)
  "Correlation Low" -> max structural rho to any co-recommended prediction < 0.4

One addition the document's own logic requires: a rule reading "Calibration >
95%" presupposes a calibration measurement EXISTS. When there is no graded
history — or the history shows an inverted ranking — no clause can pass
honestly, so the recommendation is None with the reason stated, rather than a
grade computed from unmeasured inputs.
"""
from __future__ import annotations

RULES = {
    "doc_v1": [
        # (name, level granted when ALL clauses at this level pass)
        {"level": "High", "clauses": [
            ("quality_score", ">", 90.0),
            ("calibration", ">", 0.95),
            ("variance_low", "is", True),
            ("correlation_low", "is", True),
        ]},
        # The document defines only the High rule; these graded fallbacks keep
        # the output shape total without inventing new thresholds above it.
        {"level": "Medium", "clauses": [
            ("quality_score", ">", 75.0),
            ("calibration", ">", 0.90),
            ("correlation_low", "is", True),
        ]},
        {"level": "Low", "clauses": [
            ("quality_score", ">", 60.0),
        ]},
    ],
}
DEFAULT_VERSION = "doc_v1"

VARIANCE_LOW_MIN = 0.7      # Module 2 variance component threshold
CORRELATION_LOW_MAX = 0.4   # structural rho threshold


def _facts(pred, quality: dict, corr_info: dict, assessment: dict) -> dict:
    comps = quality.get("components", {})
    return {
        "quality_score": quality.get("score"),
        "calibration": comps.get("calibration"),
        # None when variance is unmeasured — an "is True" clause then fails as
        # "unmeasured" rather than treating absence as low variance.
        "variance_low": (None if comps.get("variance") is None
                         else comps["variance"] >= VARIANCE_LOW_MIN),
        "correlation_low": corr_info.get("max_rho", 0.0) < CORRELATION_LOW_MAX,
        "evidence_signal": assessment.get("signal", "unknown"),
    }


def recommend(pred, quality: dict, corr_info: dict, assessment: dict,
              version: str = DEFAULT_VERSION) -> dict:
    """Evaluate the versioned rules for one prediction. Full audit trail."""
    facts = _facts(pred, quality, corr_info, assessment)

    # Evidence precondition: rules about calibration cannot fire when
    # calibration is unmeasured or the ranking is inverted.
    if assessment.get("signal") in ("unknown",) or assessment.get("n", 0) == 0:
        return {"recommendation": None, "rules_version": version,
                "facts": facts, "evaluated": [],
                "why": "no graded history — the rule's calibration clause has "
                       "nothing to measure against"}
    if assessment.get("signal") == "inverted":
        return {"recommendation": None, "rules_version": version,
                "facts": facts, "evaluated": [],
                "why": "confidence ranking is inverted "
                       f"({assessment['ranking']['detail']}) — recommending by "
                       "score would select the producer's worst predictions"}

    evaluated = []
    for rule in RULES[version]:
        checks = []
        passed = True
        for fact, op, want in rule["clauses"]:
            have = facts.get(fact)
            if have is None:
                ok = False
                note = "unmeasured"
            elif op == ">":
                ok = have > want
                note = f"{have} > {want}" if ok else f"{have} <= {want}"
            else:  # "is"
                ok = have is want
                note = str(have)
            checks.append({"clause": fact, "ok": ok, "detail": note})
            passed = passed and ok
        evaluated.append({"level": rule["level"], "passed": passed,
                          "checks": checks})
        if passed:
            return {"recommendation": rule["level"], "rules_version": version,
                    "facts": facts, "evaluated": evaluated,
                    "why": f"all {len(checks)} clauses of the "
                           f"{rule['level']} rule passed"}

    return {"recommendation": "None", "rules_version": version,
            "facts": facts, "evaluated": evaluated,
            "why": "no rule level had all clauses pass"}
