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

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.responses import RedirectResponse             # noqa: E402
from pydantic import BaseModel, Field                      # noqa: E402

from api import pipeline                                   # noqa: E402
from analytics import learning                             # noqa: E402
from alerts import triggers                                # noqa: E402
from ingestion import store                                # noqa: E402
from ingestion.contract import (CONF_PROBABILITY, Prediction,  # noqa: E402
                                norm_entity)
from explainability import attribution                     # noqa: E402
from scoring import gate, quality as qmod                  # noqa: E402

DB = os.environ.get("DIP_DB", os.path.join(os.path.dirname(__file__),
                                           "..", "data", "dip.sqlite3"))

# Edge-type markets whose edge is mechanical (a lock), not a calibrated forecast:
# they get de-clustered trust counting AND a separate tradeable light, and carry
# supplied tradeability attributes DIP scores but never recomputes.
TRIGGER_MARKETS = {"temp_lock"}

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
    # Supplied producer attributes for edge-type-aware markets (e.g. the weather
    # near-resolution trigger). DIP stores and scores these; it never recomputes
    # them — it has no METAR feed or order book. They ride into `raw` untouched
    # and feed the tradeable light + correlated-sample de-clustering, never the
    # outcome (which stays venue-settled).
    event_date: str | None = None    # the real event day, for (city, day) clusters
    city: str | None = None
    lag_s: float | None = None
    edge_dollars: float | None = None
    fill_200: float | None = None
    at_risk: int | None = None
    recon_delta_mean: float | None = None
    recon_delta_max: float | None = None
    recon_n: int | None = None


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
            # A supplied event_date (the real event day) wins over the push date,
            # so a lock keeps ONE identity across re-pushes and (city, day)
            # clustering is measurable. Falls back to the batch/push date.
            market=r.market, event_date=r.event_date or batch.date or r.timestamp[:10],
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


@app.get("/decision")
def get_decision(request: Request):
    """The three questions, answered off the LEDGER rather than a posted file.

    1. Is the system trusted?   -> a light per market, rolled up
    2. Is anything green today? -> a count, usually 0
    3. For each green: what, where, how much?

    Reads the ledger directly because the answer to "can I trust this" is a
    property of accumulated graded history, not of whatever batch was last
    POSTed. A page that answered it from the current batch would go green the
    moment someone uploaded a good day.
    """
    # A human typing /decision into the address bar wants the dashboard, not raw
    # JSON — browsers announce that with `Accept: text/html`, while the page's own
    # fetch() sends */*. Bounce the browser to the rendered page; keep the JSON
    # for the API (curl, the dashboard's fetch, any programmatic client).
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse("/")

    con = _con()

    # ---- graded history, grouped per market. Never pooled: one market's
    # evidence must not unlock another's recommendations. For edge-type markets
    # (the weather trigger) each row also carries the producer's SUPPLIED
    # tradeability attrs, parsed out of `raw` — DIP scores them, never recomputes.
    hist: dict[tuple, list] = {}
    for r in con.execute(
            "SELECT p.source, p.market, p.side, p.prob_over, p.event_date, "
            "p.raw, res.result "
            "FROM results res JOIN predictions p ON p.id = res.prediction_id "
            "WHERE res.result IN ('1','0')"):
        up = r["result"] == "1"
        over = r["side"] == "over"
        rec = {"market": r["market"],
               "p": r["prob_over"] if over else 1.0 - r["prob_over"],
               "hit": int(up if over else not up)}
        if r["market"] in TRIGGER_MARKETS:
            try:
                raw = json.loads(r["raw"] or "{}")
            except (ValueError, TypeError):
                raw = {}
            rec.update({
                "event_date": r["event_date"], "city": raw.get("city"),
                "fill_200": raw.get("fill_200"), "lag_s": raw.get("lag_s"),
                "at_risk": raw.get("at_risk"),
                "recon_delta_mean": raw.get("recon_delta_mean"),
                "recon_delta_max": raw.get("recon_delta_max")})
        hist.setdefault((r["source"], r["market"]), []).append(rec)

    lights, sidelined = {}, []
    for (src, mkt), rows in sorted(hist.items()):
        if mkt in TRIGGER_MARKETS:
            # De-cluster same-(city, day) locks: one weather event seen through
            # several buckets is not several independent proofs.
            clusters = Counter((row.get("city"), row.get("event_date"))
                               for row in rows)
            eff = gate.effective_independent_n(clusters.values())
            L = gate.market_light(rows, venue=src, effective_n=eff)
            L["clusters"] = len(clusters)
            # The SECOND light: is the edge actually harvestable after slippage?
            L["tradeable"] = gate.tradeable_light(rows)
            # Reliability annotation: the worst-biased settlement station here.
            biased = [row for row in rows
                      if row.get("recon_delta_max") is not None]
            if biased:
                w = max(biased, key=lambda x: abs(x["recon_delta_max"]))
                L["recon"] = {"city": w.get("city"),
                              "delta_max": w.get("recon_delta_max"),
                              "delta_mean": w.get("recon_delta_mean")}
        else:
            L = gate.market_light(rows, venue=src)
        key = f"{src}/{mkt}"
        # Distance-to-green data for the dashboard bars: current vs threshold on
        # each trust dimension. (Layer-2 edge/quality/coverage bars ride on the
        # opportunity rows, which carry those numbers already.)
        L["distance"] = {"graded": [L.get("effective_n", L["n"]), gate.MIN_GRADED],
                         "rate": [L["realized"], L["breakeven"]]}
        # A market that has not cleared its own cost of trading is set aside
        # rather than mixed in. Stated as the arithmetic, not as a hunch, and
        # self-updating: if it ever clears, it comes back on its own.
        if L["realized"] is not None and L["realized"] < L["breakeven"]:
            sidelined.append({
                "market": key, "n": L["n"],
                "why": (f"realized {L['realized']:.1%} vs {L['breakeven']:.1%} "
                        f"breakeven at {L['fee']:.0%} fee — "
                        f"{(L['breakeven']-L['realized'])*100:.1f} pts short of "
                        f"covering its own cost")})
        else:
            lights[key] = L

    if lights:
        roll = gate.rollup(lights)
    elif sidelined:
        # Every market set aside is its own answer, and a distinct one from
        # "still gathering": these markets are not unproven, they are priced
        # such that clearing their own cost is the problem.
        roll = {"light": gate.RED, "trusted": 0, "total": 0,
                "headline": (f"All {len(sidelined)} markets set aside — none "
                             f"covers its own cost of trading. Nothing to "
                             f"trust yet, and nothing worth proving here.")}
    else:
        roll = gate.rollup({})

    # ---- Layer 2: opportunities need a fair value AND a venue price for the
    # SAME question. That is the cross-source join.
    rows = con.execute(
        "SELECT domain, market, entity, event_date, COUNT(DISTINCT source) ns "
        "FROM predictions GROUP BY domain, market, entity, event_date "
        "HAVING ns > 1").fetchall()

    green, amber, blocked = [], [], []
    if not rows:
        n_pred = con.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
        srcs = [r["source"] for r in con.execute(
            "SELECT DISTINCT source FROM predictions")]
        # The honest empty state. "No opportunities" and "no opportunity is
        # EXPRESSIBLE" are different failures and must not print the same
        # sentence — the second one is a data-contract bug upstream, and
        # rendering it as a quiet market would hide it indefinitely.
        blocked.append(
            f"No model/venue pair is comparable. {n_pred} predictions from "
            f"{len(srcs)} sources ({', '.join(srcs)}), but no two share "
            f"(entity, market, event_date) — so no fair value can be priced "
            f"against a venue. Producer-side fix: the model's rows need the "
            f"same event identifier the venue uses.")

    n_open = con.execute("SELECT COUNT(*) c FROM predictions p LEFT JOIN results r "
                         "ON r.prediction_id = p.id WHERE r.prediction_id IS NULL"
                         ).fetchone()["c"]
    con.close()

    return {
        "layer1": {**roll, "markets": lights, "sidelined": sidelined},
        "layer2": {"green": green, "amber": amber, "blocked": blocked,
                   "checked": n_open, "n_markets": len(lights) + len(sidelined),
                   "message": (gate.nothing_today(n_open, len(hist))
                               if not green and not blocked else None)},
        "thresholds": {"min_graded": gate.MIN_GRADED, "min_edge": gate.MIN_EDGE,
                       "min_quality": gate.MIN_QUALITY,
                       "min_coverage": gate.MIN_COVERAGE},
    }


@app.get("/health")
def health():
    return {"status": "ok", "principle":
            "Never make predictions. Only consume them."}


@app.get("/")
def dashboard():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(os.path.dirname(__file__),
                                     "..", "dashboard", "index.html"))
