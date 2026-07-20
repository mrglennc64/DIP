"""The document's API surface, verbatim:

    POST /predictions
    GET  /quality
    GET  /portfolio
    GET  /simulation
    GET  /recommendations
    GET  /analytics
    GET  /alerts

State model: POST /predictions ingests a batch (the generic contract — any
source, any domain) into the ledger AND caches the analyzed chain for the
GET endpoints, which are views over the most recent analysis per date. The
ledger is the permanent record (Module 8); the analysis cache is derived and
rebuildable from it.

Run:  uvicorn api.main:app --port 8100        (from decision-platform/)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Query          # noqa: E402
from pydantic import BaseModel, Field                      # noqa: E402

from api import pipeline                                   # noqa: E402
from analytics import learning                             # noqa: E402
from alerts import triggers                                # noqa: E402
from ingestion import store                                # noqa: E402
from ingestion.contract import (CONF_PROBABILITY, Prediction,  # noqa: E402
                                norm_entity)
from explainability import attribution                     # noqa: E402

DB = os.environ.get("DIP_DB", os.path.join(os.path.dirname(__file__),
                                           "..", "data", "dip.sqlite3"))

app = FastAPI(title="Decision Intelligence Platform",
              description="Never makes predictions. Only consumes them.")

# date -> latest analysis. Derived state; the ledger is the record.
_analyses: dict[str, dict] = {}


def _con():
    con = store.connect(DB)
    store.init(con)
    return con


class PredictionIn(BaseModel):
    """Module 1's interface. `sport` is accepted as an alias of `domain` so the
    document's exact field name works; either satisfies the requirement."""
    id: str | None = None
    sport: str | None = None
    domain: str | None = None
    player: str | None = None
    entity: str | None = None
    market: str = "value"
    predictedValue: float | None = None
    line: float
    probabilityOver: float | None = None
    probabilityUnder: float | None = None
    confidence: float | None = None
    variance: float | None = None
    timestamp: str = ""
    source: str = "api"
    event_key: str = ""
    side: str | None = None


class PredictionBatch(BaseModel):
    predictions: list[PredictionIn]
    history: list[dict] = Field(default_factory=list,
                                description="Optional graded rows: "
                                            "{p, hit, market, ...}")
    date: str = ""


@app.post("/predictions")
def post_predictions(batch: PredictionBatch):
    """Module 1 — accept predictions from any source."""
    preds = []
    for i, r in enumerate(batch.predictions):
        display = r.player or r.entity or f"row {i}"
        preds.append(Prediction(
            source=r.source, source_version="",
            domain=r.domain or r.sport or "unspecified",
            entity=norm_entity(display), entity_display=display,
            event_key=r.event_key or f"api:{batch.date}:{i}",
            market=r.market, event_date=batch.date or r.timestamp[:10],
            line=r.line, predicted_value=r.predictedValue,
            prob_over=r.probabilityOver, prob_under=r.probabilityUnder,
            variance=r.variance, side=r.side, confidence=r.confidence,
            confidence_kind=CONF_PROBABILITY if r.confidence is not None else None,
            feature_completeness=sum(
                x is not None for x in (r.predictedValue, r.probabilityOver,
                                        r.confidence, r.variance)) / 4.0,
            raw=r.model_dump()))

    problems = {p.entity_display: p.validate() for p in preds}
    problems = {k: v for k, v in problems.items() if v}
    if problems:
        raise HTTPException(422, detail=problems)

    con = _con()
    # Sources arrive per-row (Module 1's `source` field); each must exist
    # before the FK on predictions.source will accept its rows.
    for src in {p.source for p in preds}:
        store.register_source(con, src, f"{src} (via POST /predictions)", "push")
    new, seen = store.upsert_predictions(con, preds, store.utcnow())

    history = batch.history or learning.history_rows(con)
    analysis = pipeline.analyze(preds, history, date=batch.date)
    _analyses[batch.date or "latest"] = analysis
    _analyses["latest"] = analysis          # date-less GETs read the newest
    triggers.store(con, analysis["alerts"])
    con.close()

    return {"ingested": seen, "new": new,
            "assessment": analysis["assessment"]["headline"],
            "portfolio_size": len(analysis["portfolio"]["portfolio"])}


def _analysis(date: str | None) -> dict:
    key = date or "latest"
    a = _analyses.get(key) or (_analyses.get("latest") if not date else None)
    if a is None:
        raise HTTPException(404, "no analysis yet — POST /predictions first")
    return a


@app.get("/quality")
def get_quality(date: str | None = Query(None)):
    """Module 2 — every prediction's score, components stored separately."""
    a = _analysis(date)
    return {"assessment": a["assessment"]["headline"],
            "predictions": [{
                "entity": s["pred"].entity_display,
                "market": s["pred"].market, "line": s["pred"].line,
                "quality": s["quality"],
                "explanation": s["explanation"],
            } for s in a["scored"]]}


@app.get("/portfolio")
def get_portfolio(date: str | None = Query(None)):
    """Module 3 — the top-5 selection, with why, plus the Module 4 graph."""
    a = _analysis(date)
    return {"portfolio": [{
                "entity": h["pred"].entity_display,
                "market": h["pred"].market, "line": h["pred"].line,
                "side": h["pred"].side,
                "quality": h["quality"]["score"],
                "confidence": h["pred"].confidence,
                "marginal_value": h["marginal"],
                "max_rho_at_selection": h["max_rho_at_selection"],
            } for h in a["portfolio"]["portfolio"]],
            "passed_over": a["portfolio"]["passed_over"],
            "concentration": a["portfolio"]["concentration"],
            "correlation_graph": {"nodes": a["graph"]["nodes"],
                                  "edges": a["graph"]["edges"]}}


@app.get("/simulation")
def get_simulation(date: str | None = Query(None)):
    """Module 5 — the 10,000-run answer set."""
    return _analysis(date)["simulation"]


@app.get("/recommendations")
def get_recommendations(date: str | None = Query(None)):
    """Module 9 — versioned rule evaluations, full audit trail."""
    a = _analysis(date)
    return {"recommendations": [{
        "entity": s["pred"].entity_display,
        "market": s["pred"].market, "line": s["pred"].line,
        **s["recommendation"],
    } for s in a["scored"]]}


@app.get("/analytics")
def get_analytics(domain: str | None = Query(None)):
    """Module 8 — the learning database, queried."""
    con = _con()
    out = {"by_source": learning.by_source(con, domain=domain),
           "residuals": learning.residual_summary(con),
           "ledger": store.stats(con)}
    con.close()
    return out


@app.get("/alerts")
def get_alerts(limit: int = Query(50, le=500)):
    """Module 10 — meaningful notifications, deduped."""
    con = _con()
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,))]
    con.close()
    return {"alerts": rows}


@app.get("/health")
def health():
    return {"status": "ok", "principle":
            "Never make predictions. Only consume them."}


@app.get("/")
def dashboard():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(os.path.dirname(__file__),
                                     "..", "dashboard", "index.html"))
