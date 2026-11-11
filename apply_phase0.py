#!/usr/bin/env python3
"""
Phase 0 for FinRisk Lab: infrastructure, config and database schema.

Run from the ROOT of your repo (after setup_finrisk.py):

    python apply_phase0.py            # fills the stub files
    python apply_phase0.py --force    # overwrite even if you edited them

Files that you already edited (more than a few lines) are skipped unless --force.
"""
import sys
from pathlib import Path

ROOT = Path(".").resolve()
FORCE = "--force" in sys.argv

FILES = {}

# ---------------------------------------------------------------------------
FILES["docker-compose.yml"] = '''\
# FinRisk Lab - local infrastructure.
# Phase 0: postgres + mlflow. The api and dashboard services are added in Phase 5.
name: finrisk

services:
  postgres:
    image: postgres:16
    container_name: finrisk-postgres
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-finrisk}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-finrisk}
      POSTGRES_DB: ${POSTGRES_DB:-finrisk}
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      # Init scripts run once, on first start of an empty volume, in file-name order.
      - ./docker/initdb/00_create_mlflow_db.sql:/docker-entrypoint-initdb.d/00_create_mlflow_db.sql:ro
      - ./sql/01_schema.sql:/docker-entrypoint-initdb.d/01_schema.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-finrisk} -d ${POSTGRES_DB:-finrisk}"]
      interval: 5s
      timeout: 5s
      retries: 10
    restart: unless-stopped

  mlflow:
    build:
      context: .
      dockerfile: docker/Dockerfile.mlflow
    container_name: finrisk-mlflow
    depends_on:
      postgres:
        condition: service_healthy
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri postgresql+psycopg2://${POSTGRES_USER:-finrisk}:${POSTGRES_PASSWORD:-finrisk}@postgres:5432/mlflow
      --artifacts-destination /mlflow/artifacts
      --serve-artifacts
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
    restart: unless-stopped

volumes:
  pgdata:
  mlflow_artifacts:
'''

# ---------------------------------------------------------------------------
FILES["docker/Dockerfile.mlflow"] = '''\
FROM python:3.11-slim

# Keep the major version in sync with the mlflow client in pyproject.toml
RUN pip install --no-cache-dir "mlflow>=3,<4" psycopg2-binary

EXPOSE 5000
'''

# ---------------------------------------------------------------------------
FILES["docker/initdb/00_create_mlflow_db.sql"] = '''\
-- MLflow keeps its metadata in its own database, separate from the finrisk data.
CREATE DATABASE mlflow;
'''

# ---------------------------------------------------------------------------
FILES["sql/01_schema.sql"] = '''\
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
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/config.py"] = '''\
"""Central configuration for FinRisk Lab.

All settings come from environment variables, loaded from the .env file at the
project root. Use `get_settings()` everywhere instead of reading os.environ.

Check your setup with:  python -m finrisk.config
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _get(name: str, default: str) -> str:
    return os.getenv(name, default)


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # PostgreSQL
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str
    postgres_port: int
    # MLflow
    mlflow_tracking_uri: str
    # Groq
    groq_api_key: str
    groq_model: str
    # Data
    raw_data_path: Path
    loan_term_months: int
    # Time-based split (loan issue years)
    train_start_year: int
    train_end_year: int
    valid_year: int
    oot_year: int
    monitor_year: int

    def __post_init__(self) -> None:
        if not (
            self.train_start_year
            <= self.train_end_year
            < self.valid_year
            < self.oot_year
            < self.monitor_year
        ):
            raise ValueError(
                "Split years must satisfy "
                "TRAIN_START <= TRAIN_END < VALID < OOT < MONITOR, got "
                f"{self.train_start_year}, {self.train_end_year}, "
                f"{self.valid_year}, {self.oot_year}, {self.monitor_year}"
            )
        if self.loan_term_months not in (36, 60):
            raise ValueError("LOAN_TERM_MONTHS must be 36 or 60")

    # ---- derived values ---------------------------------------------------
    @property
    def database_url(self) -> str:
        """SQLAlchemy URL for the finrisk database (used from the host machine)."""
        return (
            f"postgresql+psycopg2://{quote_plus(self.postgres_user)}:"
            f"{quote_plus(self.postgres_password)}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def sql_dir(self) -> Path:
        return PROJECT_ROOT / "sql"

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / "reports"

    @property
    def has_groq_key(self) -> bool:
        return bool(self.groq_api_key) and self.groq_api_key != "your_key_here"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    raw_path = Path(_get("RAW_DATA_PATH", "data/raw/accepted_2007_to_2018Q4.csv.gz"))
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path

    return Settings(
        postgres_user=_get("POSTGRES_USER", "finrisk"),
        postgres_password=_get("POSTGRES_PASSWORD", "finrisk"),
        postgres_db=_get("POSTGRES_DB", "finrisk"),
        postgres_host=_get("POSTGRES_HOST", "localhost"),
        postgres_port=_get_int("POSTGRES_PORT", 5432),
        mlflow_tracking_uri=_get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        groq_api_key=_get("GROQ_API_KEY", ""),
        groq_model=_get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        raw_data_path=raw_path,
        loan_term_months=_get_int("LOAN_TERM_MONTHS", 36),
        train_start_year=_get_int("TRAIN_START_YEAR", 2012),
        train_end_year=_get_int("TRAIN_END_YEAR", 2013),
        valid_year=_get_int("VALID_YEAR", 2014),
        oot_year=_get_int("OOT_YEAR", 2015),
        monitor_year=_get_int("MONITOR_YEAR", 2016),
    )


if __name__ == "__main__":
    s = get_settings()
    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Database         : {s.postgres_user}@{s.postgres_host}:{s.postgres_port}/{s.postgres_db}")
    print(f"MLflow           : {s.mlflow_tracking_uri}")
    print(f"Groq key set     : {s.has_groq_key}  (model: {s.groq_model})")
    print(f"Raw data         : {s.raw_data_path}  (exists: {s.raw_data_path.exists()})")
    print(f"Loan term        : {s.loan_term_months} months")
    print(
        f"Splits           : train {s.train_start_year}-{s.train_end_year} | "
        f"valid {s.valid_year} | oot {s.oot_year} | monitor {s.monitor_year}"
    )
'''


# ---------------------------------------------------------------------------
def write(rel: str, content: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not FORCE:
        if len(path.read_text(encoding="utf-8").splitlines()) > 3:
            print(f"  skip    {rel}  (already edited, use --force to overwrite)")
            return
    path.write_text(content, encoding="utf-8")
    print(f"  write   {rel}")


def patch(rel: str, old: str, new: str) -> None:
    path = ROOT / rel
    if not path.exists():
        print(f"  missing {rel}")
        return
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"  ok      {rel}  (already patched)")
    elif old in text:
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        print(f"  patch   {rel}")
    else:
        print(f"  warn    {rel}: expected line not found, edit it by hand")


def main() -> None:
    if not (ROOT / "src" / "finrisk").exists():
        sys.exit("Run setup_finrisk.py first, and run this from the repo root.")

    print(f"Applying Phase 0 in {ROOT}\n")
    for rel, content in FILES.items():
        write(rel, content)

    # keep the mlflow client and server on the same major version
    patch("pyproject.toml", '"mlflow",', '"mlflow>=3,<4",')
    # rebuild the mlflow image when needed
    patch("Makefile", "\tdocker compose up -d\n", "\tdocker compose up -d --build\n")

    print("\nDone. Next:")
    print("  copy .env.example to .env (if not done)")
    print("  pip install -e .")
    print("  python -m finrisk.config")
    print("  docker compose up -d --build")


if __name__ == "__main__":
    main()