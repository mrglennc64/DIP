# Decision Intelligence Platform

**Never makes predictions. Only consumes them.**

An operating system for evaluating, comparing and managing predictions from any
source, in any domain. Implements the DIP specification module for module.

## The spec's modules, where they live

| Module | Location |
|---|---|
| 1 Prediction Ingestion | `ingestion/` — contract (incl. `variance`), file/HTTP adapters, frozen ledger |
| 2 Quality Score | `scoring/quality.py` — the documented 30/30/15/15/10 weighting (`m2_v1`), plus the page-11 PQI variant (`pqi_v1`); components stored individually, coverage reported |
| 3 Portfolio Optimizer | `optimization/portfolio.py` — 30 in, top 5 out; greedy marginal selection with `(1 − max ρ)` correlation discount |
| 4 Correlation Engine | `optimization/correlation.py` — structural graph (same entity 0.8 / 0.6, same event 0.4) |
| 5 Scenario Simulator | `simulation/montecarlo.py` — 10,000 runs, Gaussian copula over the ρ matrix, best/worst/expected/distribution/sensitivity |
| 6 Explainability | `explainability/attribution.py` — signed ±points panel per prediction |
| 7 Confidence Calibration | `calibration/evidence.py` — the 4-field display (historical accuracy / model confidence / calibration error / reliability) plus curve, ranking-skill check, evidence gate |
| 8 Learning Database | `db/schema.sql` + `ingestion/store.py` (record) and `analytics/learning.py` (queries: by-source, residuals, per-version accuracy) |
| 9 Recommendation Engine | `scoring/recommendation.py` — the documented rule at the documented thresholds (`QualityScore > 90 ∧ Calibration > 95% ∧ Variance Low ∧ Correlation Low → High`), as versioned data |
| 10 Alerts | `alerts/triggers.py` — the five documented triggers, deduped per condition per day |
| APIs | `api/main.py` — `POST /predictions`, `GET /quality /portfolio /simulation /recommendations /analytics /alerts` |
| Dashboard | `dashboard/index.html`, served at `/` |

## Use

```bash
# File in -> decision out (no server, no network)
python -m api.cli predictions.csv --history graded.csv
python -m api.cli predictions.csv --explain "Judge"     # Module 6 panel

# The API + dashboard
uvicorn api.main:app --port 8100      # then open http://localhost:8100/

# Poll the owned MLB sources into the ledger (optional; DIP works without)
python -m ingestion.run ingest --date 2026-07-20
python -m ingestion.run grade  --date 2026-07-19

# Verify
python -m tests.test_modules          # module-by-module against the document
python -m tests.test_phase1           # ledger + MLB adapters
```

Files are read by column alias (`expected_ks` → predicted value, `threshold` →
line, …); the report always shows how each column was interpreted. Exit code:
0 when the portfolio is actionable on positive evidence, 2 otherwise.

## The one rule above the modules

Every scoring and recommending path is gated on evidence
(`calibration/evidence.py`). No graded history → quality is scored on structure
only and recommendations are `None` with the reason stated. Inverted confidence
ranking → a critical alert, and nothing is recommended, because ranking by
score would systematically select the producer's worst predictions. The spec's
insight is that "users benefit more from a measure of prediction reliability
than from probability alone" — which requires being honest when reliability is
unmeasured.

## Known upstream issues (routed around, fixed separately)

- `strike.perfecthold.online` odds key deactivated (failed payment) → `/v2/slate`
  500s. Legibility fix shipped in `stike/mlb-edge` (503 + reason, uncommitted).
- `fantasy.perfecthold.online/dip.json` — exporter written (`Fantasy/pick6/
  export_dip.py` + cron wiring), awaiting deploy to kv8.
- `stike/backend`: `schemas/mlb.py:32` SyntaxError; `portfolio_service.py:92`
  degenerate Monte Carlo (one shock vector for all paths — our simulator tests
  assert the opposite).
