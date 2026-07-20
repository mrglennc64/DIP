"""The full module chain, wired in the document's order.

    Ingestion -> Quality -> Portfolio -> Correlation -> Simulation ->
    Recommendation -> (Alerts)

One function, `analyze()`, runs a set of predictions plus optional graded
history through every module and returns everything each downstream surface
needs — the seven API endpoints and the file CLI are both thin views over this,
so a file on disk and a POSTed payload get identical treatment.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.adapters import file as filead              # noqa: E402
from scoring import quality, recommendation                # noqa: E402
from optimization import correlation, portfolio            # noqa: E402
from simulation import montecarlo                          # noqa: E402
from explainability import attribution                     # noqa: E402
from calibration import evidence                           # noqa: E402
from alerts import triggers                                # noqa: E402


def analyze(preds: list, history: list[dict], target_n: int = 200,
            sim_seed: int | None = 7, quality_version: str | None = None,
            date: str = "") -> dict:
    """Run the document's chain over already-adapted predictions."""
    assessment = evidence.assess(history, target_n=target_n)
    hidx = quality.index_history(history) if history else None

    # Module 2 — every prediction gets its own score.
    scored = []
    for p in preds:
        q = quality.score(p, history_index=hidx, assessment=assessment,
                          version=quality_version or quality.DEFAULT_VERSION)
        scored.append({"pred": p, "quality": q})

    # Module 4 — correlation graph (built before 3, which consumes it).
    graph = correlation.build_graph(preds)

    # Module 3 — top-5 portfolio under the three documented criteria.
    folio = portfolio.select(scored, graph)

    # Module 5 — 10k correlated simulations over the selected portfolio
    # (falls back to the full set when nothing was selected, so the
    # distribution is still answerable).
    sim_preds = [h["pred"] for h in folio["portfolio"]] or preds
    sim = montecarlo.run(sim_preds, correlation.rho_matrix(graph, sim_preds),
                         seed=sim_seed)

    # Module 6 + 9 — explanation and recommendation per prediction.
    for s in scored:
        s["explanation"] = attribution.explain(s["quality"], s["pred"])
        s["recommendation"] = recommendation.recommend(
            s["pred"], s["quality"], graph["by_id"][s["pred"].id], assessment)

    scored.sort(key=lambda s: (s["quality"]["score"] or -1), reverse=True)

    # Module 10 — alert conditions computable without the ledger.
    alerts = (triggers.highest_quality_today(scored, date)
              + triggers.calibration_drift(assessment, date))

    return {"assessment": assessment, "scored": scored, "graph": graph,
            "portfolio": folio, "simulation": sim, "alerts": alerts}


def analyze_file(pred_path: str, history_path: str | None = None,
                 domain: str = "", **kw) -> dict:
    """File in -> full chain out. The standalone flow: any CSV/JSON of
    predictions, optional graded history, no network, no domain assumptions."""
    preds, rejected, info = filead.to_predictions(pred_path, domain=domain)
    history = filead.to_history(history_path) if history_path else []
    out = analyze(preds, history, date=os.path.basename(pred_path), **kw)
    out["mapping"] = info
    out["rejected"] = rejected
    return out
