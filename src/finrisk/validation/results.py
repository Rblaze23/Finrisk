"""Validation results contract: one schema shared by the tests, dashboard, report and LLM.

Every test returns TestResult objects:
    {"model", "test", "metric", "value", "status", "warn", "fail", "rule", "details"}
status is one of: pass | warn | fail | info
Results are written to results.validation_results (Postgres) and reports/validation_results.json.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Validation policy: (warn, fail) thresholds. Documented in the validation report.
THRESHOLDS = {
    "calibration_ratio_dev": (0.10, 0.20),   # |observed/predicted - 1|
    "calibration_slope_dev": (0.10, 0.25),   # |calibration slope - 1|
    "band_underprediction": (1, 3),          # number of risk bands significantly under-predicted
    "gini_drop_relative": (0.10, 0.20),      # relative Gini loss
    "gini_floor": (0.25, 0.15),              # minimum acceptable Gini (lower is worse)
    "psi": (0.10, 0.25),                     # population stability index
}

SPLITS = ["train", "valid", "oot", "monitor"]


@dataclass
class TestResult:
    __test__ = False  # not a pytest class

    model: str
    test: str
    metric: str
    value: float | None
    status: str
    warn: float | None = None
    fail: float | None = None
    rule: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return clean(asdict(self))


@dataclass
class ValidationContext:
    """Everything a test needs about one model."""
    model_name: str                      # e.g. scorecard_independent
    kind: str                            # scorecard | xgboost | lightgbm
    role: str                            # champion | challenger
    version: str
    feature_set: str
    model: object
    splits: dict                         # split -> DataFrame (features, is_default, issue_d, ...)
    pds: dict                            # split -> np.ndarray of PDs
    numeric: list
    categorical: list

    def y(self, split: str) -> np.ndarray:
        return self.splits[split]["is_default"].to_numpy().astype(int)


def grade(value, warn: float, fail: float, higher_is_worse: bool = True) -> str:
    """pass / warn / fail from two thresholds. A value that cannot be computed is 'warn'."""
    if value is None or not np.isfinite(value):
        return "warn"
    if higher_is_worse:
        return "fail" if value > fail else "warn" if value > warn else "pass"
    return "fail" if value < fail else "warn" if value < warn else "pass"


def make(model: str, test: str, metric: str, value, status: str,
         warn=None, fail=None, rule: str = "", details: dict | None = None) -> TestResult:
    v = None if value is None or (isinstance(value, float) and not math.isfinite(value)) else float(value)
    return TestResult(model, test, metric, v, status, warn, fail, rule, details or {})


def clean(obj):
    """Make anything JSON-safe: numpy -> python, NaN/inf -> None, DataFrame -> records."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return clean(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return clean(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (pd.Timestamp, datetime, date, pd.Period)):
        return str(obj)
    return obj


def write_json(path: Path, run_id: str, results: list[TestResult], models: dict) -> None:
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "thresholds": clean({k: {"warn": v[0], "fail": v[1]} for k, v in THRESHOLDS.items()}),
        "models": clean(models),
        "results": [r.to_dict() for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_db(run_id: str, results: list[TestResult]) -> None:
    from psycopg2.extras import Json, execute_values

    from finrisk.db import connect

    rows = []
    for r in results:
        d = r.to_dict()
        details = dict(d["details"] or {})
        details.update({"warn": d["warn"], "rule": d["rule"]})
        rows.append((run_id, d["model"], d["test"], d["metric"], d["value"],
                     d["fail"], d["status"], Json(details)))
    conn = connect()
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO results.validation_results "
                "(run_id, model_name, test, metric, value, threshold, status, details) VALUES %s",
                rows)
        conn.commit()
    finally:
        conn.close()
