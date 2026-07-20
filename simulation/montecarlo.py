"""Module 5 — Scenario Simulator. "This is where Monte Carlo becomes valuable."

    Run: 10,000 simulations.
    Answer: best case, worst case, expected outcome, probability distribution,
            sensitivity to individual predictions.

Correlated Bernoulli via a Gaussian copula over the Module 4 rho matrix:
draw a correlated normal vector per simulation, hit prediction i when its
component falls below Phi^-1(p_i). This respects both each prediction's own
probability and the pairwise structure — the entire reason simulation beats
closed form here, because with correlated legs the joint distribution is not
the product of the marginals.

Implementation notes, both learned from bugs in the estate rather than theory:
- One fresh shock vector PER SIMULATION. The stike simulator drew a single
  vector shared across all paths (portfolio_service.py:92), which makes every
  path identical and the percentile bands decorative. The test asserts the
  paths differ.
- Cholesky with a diagonal fallback: a structural rho matrix can be
  non-positive-definite (A~B, B~C strongly, A,C unrelated). Jitter the diagonal
  until it factors rather than crashing or silently dropping correlation.

stdlib only (random.gauss + statistics.NormalDist) — no numpy dependency for a
module this small.
"""
from __future__ import annotations

import random
from statistics import NormalDist

NUM_SIMS = 10_000            # the document's number
_PHI_INV = NormalDist().inv_cdf


def _cholesky(m: list[list[float]]) -> list[list[float]]:
    n = len(m)
    a = [row[:] for row in m]
    for jitter in (0.0, 1e-8, 1e-4, 1e-2, 0.1):
        try:
            L = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1):
                    s = sum(L[i][k] * L[j][k] for k in range(j))
                    if i == j:
                        v = a[i][i] + jitter - s
                        if v <= 0:
                            raise ValueError
                        L[i][j] = v ** 0.5
                    else:
                        L[i][j] = (a[i][j] - s) / L[j][j]
            return L
        except (ValueError, ZeroDivisionError):
            continue
    # Last resort: independence. Reported by the caller via "correlation_used".
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def run(preds: list, rho: list[list[float]], num_sims: int = NUM_SIMS,
        seed: int | None = None) -> dict:
    """Simulate the joint outcome of a set of predictions.

    Each prediction contributes its leaned-side probability (confidence when it
    is a probability, else max(prob_over, prob_under)). Outcome per sim = number
    of hits. Seed recorded so any run is reproducible.
    """
    n = len(preds)
    if n == 0:
        return {"num_sims": 0, "note": "nothing to simulate"}

    ps = []
    for p in preds:
        if p.prob_over is not None:
            ps.append(max(p.prob_over, 1.0 - p.prob_over))
        elif p.confidence is not None:
            ps.append(p.confidence)
        else:
            ps.append(0.5)      # explicit: no probability -> coin flip, visibly

    thresholds = [_PHI_INV(min(max(x, 1e-9), 1 - 1e-9)) for x in ps]
    L = _cholesky(rho)
    independent = all(L[i][j] == (1.0 if i == j else 0.0)
                      for i in range(n) for j in range(n)) and any(
        rho[i][j] for i in range(n) for j in range(n) if i != j)

    rng = random.Random(seed)
    counts = [0] * (n + 1)
    hit_totals = [0] * n
    for _ in range(num_sims):
        z = [rng.gauss(0, 1) for _ in range(n)]      # fresh vector PER SIM
        hits = 0
        for i in range(n):
            zi = sum(L[i][k] * z[k] for k in range(i + 1))
            if zi < thresholds[i]:
                hits += 1
                hit_totals[i] += 1
        counts[hits] += 1

    dist = [{"hits": k, "p": counts[k] / num_sims} for k in range(n + 1)]
    expected = sum(k * counts[k] for k in range(n + 1)) / num_sims

    # Sensitivity to individual predictions: how much P(all hit) improves if
    # prediction i were certain — the weakest leg has the largest uplift, which
    # identifies the prediction the whole outcome leans on. Estimated from the
    # simulated conditionals so correlation is respected: P(all | i hit) comes
    # from the same paths, not an independence shortcut.
    p_all = counts[n] / num_sims
    sensitivity = []
    for i in range(n):
        p_i = hit_totals[i] / num_sims
        uplift = (p_all / p_i - p_all) if p_i > 0 else None
        sensitivity.append({
            "entity": preds[i].entity_display,
            "p_marginal": round(ps[i], 4),
            "p_simulated": round(p_i, 4),
            "p_all_uplift_if_certain": round(uplift, 4) if uplift is not None else None,
        })
    sensitivity.sort(key=lambda s: -(s["p_all_uplift_if_certain"] or 0))

    return {
        "num_sims": num_sims,
        "seed": seed,
        "n_predictions": n,
        "best_case": max(k for k in range(n + 1) if counts[k]),
        "worst_case": min(k for k in range(n + 1) if counts[k]),
        "expected": round(expected, 3),
        "p_all_hit": counts[n] / num_sims,
        "p_none_hit": counts[0] / num_sims,
        "distribution": dist,
        "sensitivity": sensitivity,
        "correlation_used": not independent,
    }
