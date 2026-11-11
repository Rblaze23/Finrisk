-- FinRisk Lab: schemas and result tables.
--
-- raw.*, staging.* and features.* tables are created by the Python pipeline
-- (ingest.py, clean.py, build.py) because their columns depend on the data.
-- The tables below are stable contracts used by validation, API and dashboard.

CREATE SCHEMA IF NOT EXISTS raw;         -- untouched source data
CREATE SCHEMA IF NOT EXISTS staging;     -- cleaned data
CREATE SCHEMA IF NOT EXISTS features;    -- model-ready features
CREATE SCHEMA IF NOT EXISTS results;     -- predictions and validation outputs
CREATE SCHEMA IF NOT EXISTS monitoring;  -- portfolio monitoring

-- One row per (loan, model, version) score.
CREATE TABLE IF NOT EXISTS results.predictions (
    loan_id        BIGINT           NOT NULL,
    model_name     TEXT             NOT NULL,
    model_version  TEXT             NOT NULL,
    split          TEXT             NOT NULL CHECK (split IN ('train', 'valid', 'oot', 'monitor')),
    pd             DOUBLE PRECISION NOT NULL CHECK (pd >= 0 AND pd <= 1),
    risk_band      TEXT,
    scored_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (loan_id, model_name, model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_model_split
    ON results.predictions (model_name, model_version, split);

-- Validation output contract:
-- {"test": "psi", "metric": "score_psi", "value": 0.28, "threshold": 0.25, "status": "fail"}
CREATE TABLE IF NOT EXISTS results.validation_results (
    id          BIGSERIAL        PRIMARY KEY,
    run_id      TEXT             NOT NULL,
    model_name  TEXT             NOT NULL,
    test        TEXT             NOT NULL,
    metric      TEXT             NOT NULL,
    value       DOUBLE PRECISION,
    threshold   DOUBLE PRECISION,
    status      TEXT             NOT NULL CHECK (status IN ('pass', 'warn', 'fail', 'info')),
    details     JSONB,
    created_at  TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_validation_run
    ON results.validation_results (run_id, model_name);

-- Monthly portfolio snapshot used by the monitoring dashboard.
CREATE TABLE IF NOT EXISTS monitoring.monthly_portfolio (
    month                  DATE             NOT NULL,
    model_name             TEXT             NOT NULL,
    n_loans                INTEGER          NOT NULL,
    n_defaults             INTEGER,
    observed_default_rate  DOUBLE PRECISION,
    mean_pd                DOUBLE PRECISION,
    score_psi              DOUBLE PRECISION,
    created_at             TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (month, model_name)
);
