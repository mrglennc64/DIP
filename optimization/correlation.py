"""Module 4 — Correlation Engine. Many predictions are not independent.

The document's example: Judge Hits, Judge Total Bases, Yankees Runs — three
predictions that look independent and move together. "Represent this as a
graph": nodes are predictions, weighted edges are the association between them,

    Prediction A ──0.82── Prediction B ──0.66── Prediction C

so the optimizer (Module 3) can avoid concentrating too much on one game or
player.

Structural edges only. Every edge comes from a shared identifier present in the
data — same entity, same event — never from a coefficient fitted on the same
handful of rows it will then judge; that number would describe the sample, not
the world. Empirical estimation is the upgrade path once the learning database
(Module 8) holds enough graded history, and it belongs behind the same evidence
gate as everything else.
"""
from __future__ import annotations

# Declared assumptions, deliberately round: these mark structural relationships,
# not measurements, and decimal places would misrepresent their provenance.
RHO_SAME_ENTITY_SAME_EVENT = 0.80   # Judge hits <-> Judge total bases
RHO_SAME_ENTITY = 0.60              # same subject, different occasions
RHO_SAME_EVENT = 0.40               # Judge total bases <-> Yankees runs


def build_graph(preds: list) -> dict:
    """The document's graph: nodes + weighted edges, plus a per-node summary.

    Returns {"nodes": [...], "edges": [...], "by_id": {pred_id: {...}}} where
    by_id carries max_rho / with / edges for each prediction — what Module 3
    consumes and what Module 9's "Correlation Low" clause reads.
    """
    nodes = [{"id": p.id, "label": p.entity_display, "market": p.market,
              "event_key": p.event_key} for p in preds]
    edges = []
    by_id = {p.id: {"max_rho": 0.0, "with": None, "edges": []} for p in preds}

    for i, a in enumerate(preds):
        for b in preds[i + 1:]:
            same_entity = a.entity == b.entity and a.domain == b.domain
            same_event = bool(a.event_key) and a.event_key == b.event_key
            if same_entity and same_event:
                rho, basis = RHO_SAME_ENTITY_SAME_EVENT, "same entity, same event"
            elif same_entity:
                rho, basis = RHO_SAME_ENTITY, "same entity"
            elif same_event:
                rho, basis = RHO_SAME_EVENT, "same event"
            else:
                continue
            edges.append({"a": a.id, "b": b.id, "rho": rho, "basis": basis})
            for x, y in ((a, b), (b, a)):
                e = by_id[x.id]
                e["edges"].append({"id": y.id, "with": y.entity_display,
                                   "rho": rho, "basis": basis})
                if rho > e["max_rho"]:
                    e["max_rho"], e["with"] = rho, y.entity_display

    return {"nodes": nodes, "edges": edges, "by_id": by_id}


def rho_between(graph: dict, id_a: str, id_b: str) -> float:
    """Pairwise rho from the graph; 0.0 when no structural edge exists."""
    for e in graph["by_id"].get(id_a, {}).get("edges", []):
        if e["id"] == id_b:
            return e["rho"]
    return 0.0


def rho_matrix(graph: dict, preds: list) -> list[list[float]]:
    """Full matrix for the simulator (Module 5). Diagonal 1.0."""
    n = len(preds)
    m = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            r = rho_between(graph, preds[i].id, preds[j].id)
            m[i][j] = m[j][i] = r
    return m


def concentration(preds: list) -> dict:
    """Herfindahl over events and entities — "too much on one game or player"
    in one number. 1.0 = everything on one; 1/n = perfectly spread."""
    def hhi(keys):
        if not keys:
            return None
        n = len(keys)
        counts: dict = {}
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
        return sum((c / n) ** 2 for c in counts.values())

    events = [p.event_key for p in preds]
    entities = [p.entity for p in preds]
    h = hhi(events)
    return {
        "n": len(preds),
        "unique_events": len(set(events)),
        "unique_entities": len(set(entities)),
        "hhi_event": h,
        "hhi_entity": hhi(entities),
        "effective_independent_bets": round(1 / h, 1) if h else None,
    }
