"""Module 3 — Portfolio Optimizer. "Instead of selecting bets, build an
optimization engine."

    Input:  30 predictions
    Output: Top 5 — highest confidence, lowest correlation, highest expected
            quality. Think like an investment portfolio.

Greedy marginal selection, which is the portfolio construction that matches the
document's three criteria directly: at each step take the candidate with the
best quality-and-confidence score AFTER discounting for correlation with what
is already held. A plain top-N-by-score ignores the third criterion and will
happily stack five predictions on one game; the pairwise discount is exactly
"avoid concentrating too much on one game or player" (Module 4) applied at
selection time.

Deliberately does not optimize stake sizes. The document scopes this module to
selection ("Top 5"); staking is a betting concern and Module 1 is explicit that
there is no betting logic in the platform.
"""
from __future__ import annotations

from . import correlation as corr_mod

TOP_N = 5                 # the document's output size
MAX_PER_EVENT = 2         # hard stop on single-game concentration


def _marginal(cand: dict, held: list[dict], graph: dict) -> float:
    """Candidate's score after correlation discount against the held set.

    Discount factor is (1 - max rho to any held pick): a 0.80-correlated
    candidate keeps 20% of its score, an uncorrelated one keeps all of it.
    Max rather than mean because risk concentrates through the single
    strongest link, not the average one.
    """
    base = cand["combined"]
    if not held:
        return base
    worst = max(corr_mod.rho_between(graph, cand["pred"].id, h["pred"].id)
                for h in held)
    return base * (1.0 - worst)


def select(scored: list[dict], graph: dict, top_n: int = TOP_N,
           max_per_event: int = MAX_PER_EVENT) -> dict:
    """scored: [{"pred": Prediction, "quality": <Module 2 dict>}, ...]

    Returns the portfolio with, for every member, WHY it was taken — and for
    every excluded candidate why it was not. An optimizer whose selections
    cannot be audited is indistinguishable from a random one.
    """
    candidates = []
    for s in scored:
        q = s["quality"]["score"]
        conf = s["pred"].confidence
        if q is None:
            continue
        # "Highest confidence" and "highest expected quality" are two of the
        # document's three criteria; equal-weight them into one base score.
        # (The third, lowest correlation, is applied marginally in _marginal.)
        combined = (q / 100.0) * 0.5 + (conf if conf is not None else 0.5) * 0.5
        candidates.append({**s, "combined": combined})

    held, passed_over = [], []
    pool = sorted(candidates, key=lambda c: -c["combined"])

    while pool and len(held) < top_n:
        best, best_val = None, -1.0
        for c in pool:
            ev = c["pred"].event_key
            if sum(1 for h in held if h["pred"].event_key == ev) >= max_per_event:
                continue
            v = _marginal(c, held, graph)
            if v > best_val:
                best, best_val = c, v
        if best is None:
            break
        pool.remove(best)
        max_rho = (max((corr_mod.rho_between(graph, best["pred"].id, h["pred"].id)
                        for h in held), default=0.0))
        held.append({**best, "marginal": round(best_val, 4),
                     "max_rho_at_selection": max_rho})

    for c in pool:
        ev = c["pred"].event_key
        if sum(1 for h in held if h["pred"].event_key == ev) >= max_per_event:
            why = f"event cap: already holding {max_per_event} from {ev}"
        elif len(held) >= top_n:
            why = f"portfolio full at {top_n}"
        else:
            why = "lower marginal value than every selection"
        passed_over.append({"entity": c["pred"].entity_display,
                            "combined": round(c["combined"], 4), "why": why})

    return {
        "portfolio": held,
        "passed_over": passed_over,
        "concentration": corr_mod.concentration([h["pred"] for h in held]),
        "criteria": {"top_n": top_n, "max_per_event": max_per_event,
                     "selection": "greedy marginal quality*confidence with "
                                  "(1 - max rho) correlation discount"},
    }
