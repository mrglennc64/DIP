-- Decision Intelligence Platform — canonical ledger.
--
-- The one thing DIP owns that no upstream does: every prediction from every
-- source, in one shape, joinable, with its outcome attached. Today those
-- records live in three places that cannot be compared — Fantasy's
-- predictions_log.csv, mlb-edge's predictions.csv, and stike's `predictions`
-- table (which has no outcome column at all). This schema is the merge.
--
-- Deliberately NOT one wide table (the spec is explicit): predictions never
-- carry outcomes. A prediction is what was believed before the event; a result
-- is what happened. Keeping them in one row invites the update that quietly
-- rewrites history, which is the failure that makes an accuracy record
-- worthless. Results are append-only against a prediction id.

PRAGMA journal_mode = WAL;      -- hourly writer + dashboard readers, no blocking
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------- sources ---
CREATE TABLE IF NOT EXISTS prediction_sources (
    source        TEXT PRIMARY KEY,       -- "mlbedge" | "fantasy" | ...
    display_name  TEXT NOT NULL,
    kind          TEXT NOT NULL,          -- "http_api" | "http_json" | "push"
    endpoint      TEXT,
    notes         TEXT
);

-- Every ingest attempt, success or failure. This is what answers "is a source
-- stale?" — the question you cannot ask from the predictions table, because a
-- source that silently stops returning rows looks identical to a quiet day.
CREATE TABLE IF NOT EXISTS ingest_runs (
    id            INTEGER PRIMARY KEY,
    source        TEXT NOT NULL REFERENCES prediction_sources(source),
    event_date    TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL,          -- "ok" | "empty" | "error"
    rows_seen     INTEGER NOT NULL DEFAULT 0,
    rows_new      INTEGER NOT NULL DEFAULT 0,
    rows_updated  INTEGER NOT NULL DEFAULT 0,
    upstream_stamp TEXT,                  -- source's own freshness marker
    error         TEXT
);
CREATE INDEX IF NOT EXISTS ix_ingest_runs_source_date
    ON ingest_runs(source, event_date, started_at DESC);

-- ------------------------------------------------------------ predictions ---
-- One row per stable prediction identity. `id` = hash(source, domain, entity,
-- market, line, event_date) — a moved line is a genuinely different prediction,
-- so it gets its own id rather than overwriting the one that was already made.
--
-- This row is FROZEN at first sight. Upstreams re-project during the day
-- (mlb-edge's /v2/slate recomputes past dates with current stats), and letting
-- a later poll overwrite the original would leak outcome information backwards
-- into what we claim was predicted beforehand. Later polls land in
-- prediction_observations instead.
CREATE TABLE IF NOT EXISTS predictions (
    id             TEXT PRIMARY KEY,
    source         TEXT NOT NULL REFERENCES prediction_sources(source),
    source_version TEXT NOT NULL DEFAULT '',   -- mu_version / model build
    domain         TEXT NOT NULL,              -- "MLB"
    entity         TEXT NOT NULL,              -- accent-folded lowercase
    entity_display TEXT NOT NULL,
    event_key      TEXT NOT NULL,              -- game — structural correlation joins here
    market         TEXT NOT NULL,
    event_date     TEXT NOT NULL,

    predicted_value      REAL,
    line                 REAL NOT NULL,
    prob_over            REAL,
    prob_under           REAL,
    prob_uncalibrated    REAL,                 -- pre-calibration p; measures the calibrator's own lift
    variance             REAL,                 -- Module 1 field; 15% of the Module 2 weighting
    dispersion           REAL,                 -- NB size param; NULL = Poisson assumed
    side                 TEXT,
    confidence           REAL,                 -- normalized 0-1
    confidence_kind      TEXT,                 -- what it MEANT upstream; see note below
    feature_completeness REAL,

    first_seen_at  TEXT NOT NULL,
    raw            TEXT NOT NULL               -- full upstream row, never discarded
);
-- confidence_kind exists because the two sources mean different things by the
-- word. mlb-edge returns a categorical High/Medium/Low; stike's numeric one is
-- abs(p - 0.5) * 100, i.e. distance-from-even, which is not an uncertainty
-- estimate at all. Normalizing both to 0-1 without recording which is which
-- would fabricate comparability, so PQI reads this column and refuses to pool
-- kinds it cannot compare.

CREATE INDEX IF NOT EXISTS ix_pred_date       ON predictions(event_date);
CREATE INDEX IF NOT EXISTS ix_pred_entity     ON predictions(domain, entity, event_date);
CREATE INDEX IF NOT EXISTS ix_pred_event      ON predictions(event_key, event_date);
CREATE INDEX IF NOT EXISTS ix_pred_source     ON predictions(source, event_date);

-- Append-only: what each poll saw for an already-known prediction. Two jobs:
--   1. PQI's "prediction stability" component — the within-day variance of
--      predicted_value. DIP can measure this only because it polls hourly and
--      keeps every reading; the sources themselves retain just the latest.
--   2. An audit trail proving the frozen row was never edited.
CREATE TABLE IF NOT EXISTS prediction_observations (
    id              INTEGER PRIMARY KEY,
    prediction_id   TEXT NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    observed_at     TEXT NOT NULL,
    predicted_value REAL,
    prob_over       REAL,
    confidence      REAL,
    line            REAL,
    UNIQUE(prediction_id, observed_at)
);
CREATE INDEX IF NOT EXISTS ix_obs_pred ON prediction_observations(prediction_id);

-- ---------------------------------------------------------------- results ---
-- result codes follow Fantasy's grade.py: '1' hit, '0' miss, 'X' push (the stat
-- landed exactly on a whole line). 'X' is NOT a loss — grade.py excludes it
-- from hit rates, and collapsing it would bias every accuracy number DIP
-- derives. Stored as TEXT so the push stays a first-class value.
CREATE TABLE IF NOT EXISTS results (
    prediction_id TEXT PRIMARY KEY REFERENCES predictions(id) ON DELETE CASCADE,
    actual        REAL,
    result        TEXT NOT NULL CHECK (result IN ('1', '0', 'X')),
    residual      REAL,                    -- actual - predicted_value
    settled_at    TEXT NOT NULL,
    graded_by     TEXT NOT NULL            -- "fantasy_export" | "statsapi" | ...
);
CREATE INDEX IF NOT EXISTS ix_results_result ON results(result);

-- Canonical game resolution: (domain, entity, event_date) -> real event id.
--
-- The sources do not agree on what a game is called. Fantasy carries a board
-- label ("TEX@HOU"); mlb-edge carries opponent + venue and no game id at all.
-- Joining structural correlation on either would silently fail across sources —
-- two legs in the same game would look independent, and the optimizer would
-- concentrate a whole card on one game while reporting the correlation as low.
--
-- So the source's own label stays on the prediction row (never edited — the row
-- is frozen), and the authoritative key lives here, resolved from the MLB
-- StatsAPI schedule during grading, which already walks exactly that data.
-- Correlation joins THROUGH this table rather than on the raw label.
CREATE TABLE IF NOT EXISTS entity_events (
    domain          TEXT NOT NULL,
    entity          TEXT NOT NULL,
    event_date      TEXT NOT NULL,
    canonical_event TEXT NOT NULL,        -- e.g. "mlb:745812" (gamePk)
    resolved_at     TEXT NOT NULL,
    resolved_by     TEXT NOT NULL,
    PRIMARY KEY (domain, entity, event_date)
);
CREATE INDEX IF NOT EXISTS ix_entity_events_canon
    ON entity_events(canonical_event, event_date);

-- ------------------------------------------------------ quality (phase 2) ---
CREATE TABLE IF NOT EXISTS calibration_history (
    id             INTEGER PRIMARY KEY,
    source         TEXT NOT NULL REFERENCES prediction_sources(source),
    market         TEXT NOT NULL,
    as_of          TEXT NOT NULL,
    n              INTEGER,
    brier          REAL,
    log_loss       REAL,
    stated_rate    REAL,
    realized_rate  REAL,
    gap_pts        REAL,
    mae            REAL,
    bias           REAL,
    buckets        TEXT,                   -- JSON reliability curve
    UNIQUE(source, market, as_of)
);

-- One row per prediction PER PQI VERSION. A refit adds rows; it never rewrites
-- a score that was already shown. Components are stored individually because
-- the v1 weights are an unvalidated placeholder from the spec — the whole point
-- is being able to refit them out-of-sample later against these very columns.
CREATE TABLE IF NOT EXISTS quality_scores (
    prediction_id        TEXT NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    pqi_version          TEXT NOT NULL,
    computed_at          TEXT NOT NULL,
    score                REAL NOT NULL,
    c_calibration        REAL,
    c_confidence         REAL,
    c_completeness       REAL,
    c_stability          REAL,
    c_recent_performance REAL,
    weights              TEXT NOT NULL,    -- JSON, so a score is reproducible from its row
    PRIMARY KEY (prediction_id, pqi_version)
);

CREATE TABLE IF NOT EXISTS model_versions (
    source        TEXT NOT NULL REFERENCES prediction_sources(source),
    version       TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL,
    PRIMARY KEY (source, version)
);

-- ----------------------------------------------------- decision (phase 3) ---
CREATE TABLE IF NOT EXISTS correlations (
    event_date  TEXT NOT NULL,
    id_a        TEXT NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    id_b        TEXT NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    rho         REAL NOT NULL,
    basis       TEXT NOT NULL,            -- "same_event" | "same_entity" | "day_factor"
    PRIMARY KEY (event_date, id_a, id_b)
);

CREATE TABLE IF NOT EXISTS simulation_runs (
    id            INTEGER PRIMARY KEY,
    event_date    TEXT NOT NULL,
    ran_at        TEXT NOT NULL,
    num_sims      INTEGER NOT NULL,
    portfolio_id  INTEGER,
    summary       TEXT NOT NULL,          -- JSON: best/worst/expected/distribution/sensitivity
    seed          INTEGER                 -- recorded so a run is reproducible
);

CREATE TABLE IF NOT EXISTS portfolios (
    id           INTEGER PRIMARY KEY,
    event_date   TEXT NOT NULL,
    built_at     TEXT NOT NULL,
    strategy     TEXT NOT NULL,
    constraints  TEXT NOT NULL,           -- JSON
    members      TEXT NOT NULL            -- JSON list of prediction ids
);

-- Rules live as versioned DATA, not code, so a recommendation policy can change
-- without touching anything on the prediction path.
CREATE TABLE IF NOT EXISTS recommendations (
    prediction_id TEXT NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    rule_version  TEXT NOT NULL,
    computed_at   TEXT NOT NULL,
    level         TEXT NOT NULL,          -- "High" | "Medium" | "Low" | "None"
    rationale     TEXT NOT NULL,          -- JSON: which clauses fired
    PRIMARY KEY (prediction_id, rule_version)
);

CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY,
    created_at  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    severity    TEXT NOT NULL,
    subject     TEXT NOT NULL,
    body        TEXT NOT NULL,
    dedupe_key  TEXT NOT NULL UNIQUE,     -- one alert per condition per day, not per poll
    delivered_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_alerts_created ON alerts(created_at DESC);
