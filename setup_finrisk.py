#!/usr/bin/env python3
"""
Scaffold generator for FinRisk Lab.

Run it from INSIDE your cloned repo folder (the one open in VS Code):

    python setup_finrisk.py            # builds the structure in the current folder
    python setup_finrisk.py some/dir   # or in another folder

Existing files are never overwritten, so it is safe to re-run.
"""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

# ---------------------------------------------------------------------------
# Python modules: path (relative to src/finrisk) -> one-line purpose
# ---------------------------------------------------------------------------
MODULES = {
    "config.py": "Central configuration: paths, DB URL, split years, Groq settings (loaded from .env).",
    # data
    "data/ingest.py": "Load LendingClub CSV into PostgreSQL raw schema (supports --sample).",
    "data/clean.py": "Clean raw loans into the staging schema and build the default target.",
    "data/splits.py": "Time-based train / validation / out-of-time splits. Single source of truth.",
    # features
    "features/woe.py": "Binning and Weight of Evidence / Information Value transformer.",
    "features/build.py": "Feature engineering pipeline writing to the features schema.",
    # models
    "models/scorecard.py": "WoE + logistic regression scorecard (regulated-style baseline).",
    "models/gbm.py": "XGBoost and LightGBM model definitions.",
    "models/train.py": "MLflow-tracked training for all models.",
    "models/registry.py": "Champion / challenger promotion using the MLflow model registry.",
    # explain
    "explain/shap_explainer.py": "SHAP global and local explanations.",
    "explain/reason_codes.py": "Convert SHAP values into adverse-action style reason codes.",
    # validation
    "validation/discrimination.py": "AUC, Gini, KS.",
    "validation/calibration.py": "Brier score, reliability curve, calibration tests, recalibration.",
    "validation/stability.py": "PSI and CSI.",
    "validation/leakage.py": "Leakage tests: post-origination variables and their impact.",
    "validation/out_of_time.py": "Performance degradation from train to out-of-time sample.",
    "validation/explanation_stability.py": "Stability of SHAP explanations across time periods.",
    "validation/fairness.py": "Simple fairness and sensitivity checks.",
    "validation/run_validation.py": "Orchestrator: runs all tests and writes the validation results JSON.",
    # monitoring
    "monitoring/drift.py": "Portfolio drift metrics: default rate, score distribution, PSI over time.",
    "monitoring/simulate.py": "Monthly portfolio monitoring simulation.",
    # genai
    "genai/groq_client.py": "Groq SDK wrapper with retry, backoff and caching.",
    "genai/report_writer.py": "LLM drafts validation report sections from the metrics JSON only.",
    "genai/number_guard.py": "Verifies every figure in LLM output exists in the source metrics.",
    "genai/rag.py": "RAG over regulatory documents (SR 11-7, EBA, IFRS 9).",
    # api
    "api/main.py": "FastAPI app: /score, /health, /model-info.",
    "api/schemas.py": "Pydantic request and response schemas.",
    # dashboard
    "dashboard/app.py": "Streamlit dashboard: scoring, validation results, monitoring.",
}

PACKAGES = ["", "data", "features", "models", "explain", "validation",
            "monitoring", "genai", "api", "dashboard"]

SQL_FILES = {
    "01_schema.sql": "Create schemas: raw, staging, features, results, monitoring.",
    "02_staging.sql": "Clean raw.loans into staging.loans_clean.",
    "03_features.sql": "Feature aggregations into features.loan_features.",
    "04_monitoring_views.sql": "Views: default rate by vintage, score band migration.",
}

TESTS = {
    "test_stability.py": "Unit tests for PSI / CSI.",
    "test_calibration.py": "Unit tests for calibration metrics.",
    "test_woe.py": "Unit tests for WoE / IV.",
    "test_number_guard.py": "Unit tests for the LLM number verification guard.",
}

# ---------------------------------------------------------------------------
# Static file contents
# ---------------------------------------------------------------------------
GITIGNORE = """\
# Python
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/

# Secrets (never commit: the repo is public)
.env

# Data and artifacts (the Kaggle dataset must not be re-published)
data/raw/*
!data/raw/.gitkeep
mlruns/
mlflow.db
*.parquet
*.pkl
*.csv
*.csv.gz

# OS / IDE
.DS_Store
.vscode/
.idea/
"""

ENV_EXAMPLE = """\
# --- PostgreSQL ---
POSTGRES_USER=finrisk
POSTGRES_PASSWORD=finrisk
POSTGRES_DB=finrisk
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# --- MLflow ---
MLFLOW_TRACKING_URI=http://localhost:5000

# --- Groq (free API key from console.groq.com) ---
GROQ_API_KEY=your_key_here
# Check Groq's current model list; names change over time
GROQ_MODEL=llama-3.3-70b-versatile

# --- Data ---
RAW_DATA_PATH=data/raw/accepted_2007_to_2018Q4.csv.gz
LOAN_TERM_MONTHS=36

# --- Time-based split (loan issue years) ---
# 36-month loans only, so outcomes are fully observed
TRAIN_START_YEAR=2012
TRAIN_END_YEAR=2013
VALID_YEAR=2014
OOT_YEAR=2015
MONITOR_YEAR=2016
"""

MAKEFILE = (
    ".PHONY: up down ingest train validate test\n\n"
    "up:\n\tdocker compose up -d\n\n"
    "down:\n\tdocker compose down\n\n"
    "ingest:\n\tpython -m finrisk.data.ingest --sample\n\n"
    "train:\n\tpython -m finrisk.models.train\n\n"
    "validate:\n\tpython -m finrisk.validation.run_validation\n\n"
    "test:\n\tpytest -q\n"
)

PYPROJECT = """\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "finrisk"
version = "0.1.0"
description = "Credit risk intelligence and model validation platform"
requires-python = ">=3.10"
dependencies = [
    "pandas", "numpy", "scikit-learn", "xgboost", "lightgbm", "shap",
    "mlflow", "fastapi", "uvicorn", "streamlit", "psycopg2-binary",
    "sqlalchemy", "python-dotenv", "pydantic", "scipy", "matplotlib",
    "groq", "pytest",
]

[tool.setuptools.packages.find]
where = ["src"]
"""

README = """\
# FinRisk Lab

Credit Risk Intelligence and Model Validation Platform.

> Status: work in progress. This README is completed as the project progresses.

## Quick start
```bash
cp .env.example .env
pip install -e .
make up
```

## Data
LendingClub accepted loans 2007-2018Q4 (Kaggle: `wordsforthewise/lending-club`).
Download `accepted_2007_to_2018Q4.csv.gz` into `data/raw/`. The data is not
stored in this repository.
"""

DOCKER_STUBS = {
    "Dockerfile.api": "# TODO: API image\n",
    "Dockerfile.dashboard": "# TODO: dashboard image\n",
}


# ---------------------------------------------------------------------------
def write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rel = path.relative_to(ROOT)
    if path.exists():
        print(f"  skip   {rel}")
        return
    path.write_text(content, encoding="utf-8")
    print(f"  create {rel}")


def py_stub(description: str) -> str:
    return f'"""{description}"""\n'


def main() -> None:
    print(f"Creating FinRisk Lab scaffold in: {ROOT}\n")

    # top-level files
    write(ROOT / "README.md", README)
    write(ROOT / "pyproject.toml", PYPROJECT)
    write(ROOT / "Makefile", MAKEFILE)
    write(ROOT / "docker-compose.yml", "# TODO: postgres, mlflow, api, dashboard\n")
    write(ROOT / ".env.example", ENV_EXAMPLE)
    write(ROOT / ".gitignore", GITIGNORE)

    # data folders
    write(ROOT / "data/raw/.gitkeep")
    write(ROOT / "data/regulatory_docs/.gitkeep")
    write(ROOT / "reports/.gitkeep")

    # sql
    for name, desc in SQL_FILES.items():
        write(ROOT / "sql" / name, f"-- {desc}\n")

    # package __init__ files
    for pkg in PACKAGES:
        write(ROOT / "src/finrisk" / pkg / "__init__.py")

    # modules
    for rel, desc in MODULES.items():
        write(ROOT / "src/finrisk" / rel, py_stub(desc))

    # tests
    write(ROOT / "tests/__init__.py")
    for name, desc in TESTS.items():
        write(ROOT / "tests" / name, py_stub(desc))

    # docker
    for name, content in DOCKER_STUBS.items():
        write(ROOT / "docker" / name, content)

    # notebook
    nb = {
        "cells": [{"cell_type": "markdown", "metadata": {},
                   "source": ["# 01 - Exploratory Data Analysis"]}],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }
    write(ROOT / "notebooks/01_eda.ipynb", json.dumps(nb, indent=1))

    print("\nDone. Next steps:")
    print("  1. python -m venv .venv  and activate it")
    print("  2. copy .env.example to .env and add your GROQ_API_KEY")
    print("  3. put accepted_2007_to_2018Q4.csv.gz in data/raw/")
    print("  4. git add . && git commit -m 'Add project scaffold' && git push")


if __name__ == "__main__":
    main()