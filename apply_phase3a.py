#!/usr/bin/env python3
"""
Phase 3a for FinRisk Lab: the validation suite (calibration, out-of-time, stability)
plus the results contract and the orchestrator.

Run from the ROOT of your repo (after Phase 2b):

    python apply_phase3a.py            # fills the stub files
    python apply_phase3a.py --force    # overwrite even if you edited them
"""
import sys
from pathlib import Path

ROOT = Path(".").resolve()
FORCE = "--force" in sys.argv

FILES = {}

# ---------------------------------------------------------------------------
FILES["src/finrisk/validation/results.py"] = r'''"""Validation results contract: one schema shared by the tests, dashboard, report and LLM.

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
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/validation/calibration.py"] = r'''"""Calibration: do predicted PDs match observed default rates?

A model can rank borrowers well (high Gini) and still be badly calibrated. Calibration is
what matters for pricing, provisioning (IFRS 9) and capital.

  calibration_table   : decile table with Wilson confidence intervals
  hosmer_lemeshow     : chi-square calibration test (informational: always rejects with huge n)
  calibration_slope_intercept : logistic recalibration fit (ideal: slope 1, intercept 0)
  band_calibration    : per risk band, one-sided test for UNDER-prediction (the regulator's worry)
  Recalibrator        : intercept shift, Platt scaling, isotonic regression
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2, norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from finrisk.models.risk_bands import assign_risk_band
from finrisk.validation.discrimination import auc_gini_ks
from finrisk.validation.results import THRESHOLDS, grade, make

EPS = 1e-6


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def _logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def wilson_interval(k, n, z: float = 1.96):
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of predicted PD: predicted vs observed with a Wilson interval."""
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "p": _clip(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    g = (df.groupby("bin")
           .agg(n=("y", "size"), n_defaults=("y", "sum"),
                mean_pd=("p", "mean"), observed_dr=("y", "mean"))
           .reset_index())
    g["ratio"] = g["observed_dr"] / g["mean_pd"]
    lo, hi = wilson_interval(g["n_defaults"], g["n"])
    g["ci_low"], g["ci_high"] = lo, hi
    g["pd_inside_ci"] = (g["mean_pd"] >= lo) & (g["mean_pd"] <= hi)
    return g


def hosmer_lemeshow(y, p, groups: int = 10) -> dict:
    t = calibration_table(y, p, groups)
    exp = t["n"] * t["mean_pd"]
    stat = float((((t["n_defaults"] - exp) ** 2) / (exp * (1 - t["mean_pd"]))).sum())
    dof = groups - 2
    return {"statistic": stat, "dof": dof, "p_value": float(chi2.sf(stat, dof))}


def calibration_slope_intercept(y, p) -> dict:
    """Fit logit(P(default)) = intercept + slope * logit(PD). Ideal: intercept 0, slope 1."""
    x = _logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(x, np.asarray(y).astype(int))
    return {"slope": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0])}


def intercept_shift(y, p) -> float:
    """Constant log-odds shift delta so that mean PD equals the observed default rate."""
    target = float(np.mean(y))
    lg = _logit(p)
    f = lambda d: float(np.mean(_sigmoid(lg + d)) - target)
    if f(-6) * f(6) > 0:
        return float("nan")
    return float(brentq(f, -6, 6))


def band_calibration(y, p, min_n: int = 30) -> pd.DataFrame:
    """Per master-scale risk band, one-sided test: is observed > predicted (under-prediction)?"""
    y = np.asarray(y).astype(int)
    p = _clip(p)
    df = pd.DataFrame({"y": y, "p": p, "band": assign_risk_band(p)})
    rows = []
    for band, g in df.groupby("band"):
        if len(g) < min_n:
            continue
        expected = g["p"].sum()
        var = (g["p"] * (1 - g["p"])).sum()
        z = (g["y"].sum() - expected) / np.sqrt(var)
        rows.append({"band": band, "n": len(g), "mean_pd": g["p"].mean(),
                     "observed_dr": g["y"].mean(), "ratio": g["y"].mean() / g["p"].mean(),
                     "z": float(z), "p_value_under": float(norm.sf(z))})
    return pd.DataFrame(rows)


class Recalibrator:
    """Fit on one sample (e.g. validation), apply to another (e.g. out-of-time)."""

    METHODS = ("intercept", "platt", "isotonic")

    def __init__(self, method: str):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.method = method

    def fit(self, p, y) -> "Recalibrator":
        y = np.asarray(y).astype(int)
        if self.method == "intercept":
            self.delta_ = intercept_shift(y, p)
        elif self.method == "platt":
            self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        else:
            self.iso_ = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(_clip(p), y)
        return self

    def transform(self, p) -> np.ndarray:
        if self.method == "intercept":
            return _clip(_sigmoid(_logit(p) + self.delta_))
        if self.method == "platt":
            return _clip(self.lr_.predict_proba(_logit(p).reshape(-1, 1))[:, 1])
        return _clip(self.iso_.predict(_clip(p)))


def calibration_summary(y, p) -> dict:
    y = np.asarray(y).astype(int)
    p = _clip(p)
    out = {"mean_pd": float(p.mean()), "observed_dr": float(y.mean()),
           "ratio": float(y.mean() / p.mean()),
           "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p))}
    out.update(calibration_slope_intercept(y, p))
    hl = hosmer_lemeshow(y, p)
    out["hl_statistic"], out["hl_p_value"] = hl["statistic"], hl["p_value"]
    return out


# ---------------------------------------------------------------------------
def run(ctx) -> list:
    out, m = [], ctx.model_name
    w, f = THRESHOLDS["calibration_ratio_dev"]
    rule_ratio = f"|observed/predicted - 1| <= {w:.0%} pass, <= {f:.0%} warn, else fail"

    summ = {s: calibration_summary(ctx.y(s), ctx.pds[s]) for s in ["train", "valid", "oot"]}

    out.append(make(m, "calibration", "train_calibration_ratio", summ["train"]["ratio"], "info",
                    details=summ["train"]))
    for s in ["valid", "oot"]:
        sm = summ[s]
        dev = abs(sm["ratio"] - 1)
        det = dict(sm, deviation=dev)
        if s == "oot":
            det["deciles"] = calibration_table(ctx.y(s), ctx.pds[s])
        out.append(make(m, "calibration", f"{s}_calibration_ratio", sm["ratio"],
                        grade(dev, w, f), w, f, rule_ratio, det))

    ws, fs_ = THRESHOLDS["calibration_slope_dev"]
    for s in ["valid", "oot"]:
        dev = abs(summ[s]["slope"] - 1)
        out.append(make(m, "calibration", f"{s}_calibration_slope", summ[s]["slope"],
                        grade(dev, ws, fs_), ws, fs_,
                        f"|slope - 1| <= {ws} pass, <= {fs_} warn, else fail",
                        {"intercept": summ[s]["intercept"]}))

    out.append(make(m, "calibration", "oot_hosmer_lemeshow_p", summ["oot"]["hl_p_value"], "info",
                    rule="informational: with hundreds of thousands of loans the test rejects "
                         "even tiny deviations",
                    details={"statistic": summ["oot"]["hl_statistic"]}))
    out.append(make(m, "calibration", "oot_brier", summ["oot"]["brier"], "info",
                    details={"log_loss": summ["oot"]["log_loss"]}))

    bands = band_calibration(ctx.y("oot"), ctx.pds["oot"])
    n_under = int(((bands["p_value_under"] < 0.01) & (bands["ratio"] > 1)).sum())
    wb, fb = THRESHOLDS["band_underprediction"]
    out.append(make(m, "calibration", "oot_bands_underpredicted", n_under,
                    grade(n_under, wb - 0.5, fb - 0.5), wb, fb,
                    f"risk bands where observed > predicted at p<0.01: >= {wb} warn, >= {fb} fail",
                    {"bands": bands}))

    # can a static recalibration fitted on 2014 repair 2015?
    yv, pv = ctx.y("valid"), ctx.pds["valid"]
    yo, po = ctx.y("oot"), ctx.pds["oot"]
    base = summ["oot"]
    for method in Recalibrator.METHODS:
        p2 = Recalibrator(method).fit(pv, yv).transform(po)
        ratio2 = float(yo.mean() / p2.mean())
        out.append(make(
            m, "recalibration", f"{method}_oot_calibration_ratio", ratio2,
            grade(abs(ratio2 - 1), w, f), w, f,
            f"recalibrated on valid (2014), evaluated on oot (2015). " + rule_ratio,
            {"ratio_before": base["ratio"], "brier_before": base["brier"],
             "brier_after": float(brier_score_loss(yo, p2)),
             "gini_before": auc_gini_ks(yo, po)["gini"],
             "gini_after": auc_gini_ks(yo, p2)["gini"]}))
    return out
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/validation/stability.py"] = r'''"""Population Stability Index (PSI) and Characteristic Stability Index (CSI).

PSI = sum( (actual% - expected%) * ln(actual% / expected%) )  over bins.
  < 0.10  stable   |   0.10 - 0.25  moderate shift   |   > 0.25  significant shift

  Score PSI : shift of the model's PD distribution vs the training population
  CSI       : the same measure per input feature (missing values form their own bin)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finrisk.validation.results import THRESHOLDS, grade, make

EPS = 1e-4
MISSING = "__MISSING__"


def psi_label(v: float) -> str:
    w, f = THRESHOLDS["psi"]
    return "stable" if v <= w else "moderate shift" if v <= f else "significant shift"


def _psi_from_counts(expected, actual) -> float:
    e, a = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    if e.sum() == 0 or a.sum() == 0:
        return float("nan")
    ep = np.clip(e / e.sum(), EPS, None)
    ap = np.clip(a / a.sum(), EPS, None)
    return float(np.sum((ap - ep) * np.log(ap / ep)))


def psi_numeric(expected, actual, n_bins: int = 10) -> float:
    """Bins are deciles of the EXPECTED (training) sample; missing values get their own bin."""
    e = pd.to_numeric(pd.Series(expected), errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(pd.Series(actual), errors="coerce").to_numpy(dtype=float)
    ev = e[~np.isnan(e)]
    if ev.size == 0 or np.unique(ev).size < 2:
        edges = np.array([], dtype=float)
    else:
        edges = np.unique(np.quantile(ev, np.linspace(0, 1, n_bins + 1)[1:-1]))

    def counts(x):
        miss = np.isnan(x)
        idx = np.searchsorted(edges, x[~miss], side="left")
        c = np.bincount(idx, minlength=len(edges) + 1).astype(float)
        return np.append(c, miss.sum())

    return _psi_from_counts(counts(e), counts(a))


def psi_categorical(expected, actual) -> float:
    def prep(x):
        s = pd.Series(x)
        return s.where(s.notna(), MISSING).astype(str)

    e, a = prep(expected), prep(actual)
    levels = sorted(set(e) | set(a))
    ce = e.value_counts().reindex(levels, fill_value=0)
    ca = a.value_counts().reindex(levels, fill_value=0)
    return _psi_from_counts(ce.to_numpy(), ca.to_numpy())


def csi_table(expected_df: pd.DataFrame, actual_df: pd.DataFrame,
              numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    rows = []
    for c in numeric:
        rows.append({"feature": c, "kind": "numeric",
                     "psi": psi_numeric(expected_df[c], actual_df[c]),
                     "missing_expected": float(expected_df[c].isna().mean()),
                     "missing_actual": float(actual_df[c].isna().mean())})
    for c in categorical:
        rows.append({"feature": c, "kind": "categorical",
                     "psi": psi_categorical(expected_df[c], actual_df[c]),
                     "missing_expected": float(expected_df[c].isna().mean()),
                     "missing_actual": float(actual_df[c].isna().mean())})
    t = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    t["label"] = t["psi"].map(psi_label)
    return t


# ---------------------------------------------------------------------------
def run(ctx) -> list:
    out, m = [], ctx.model_name
    w, f = THRESHOLDS["psi"]
    rule = f"PSI <= {w} pass, <= {f} warn, else fail"

    for s in ["valid", "oot", "monitor"]:
        v = psi_numeric(ctx.pds["train"], ctx.pds[s])
        out.append(make(m, "stability", f"score_psi_train_to_{s}", v, grade(v, w, f), w, f, rule))

    train = ctx.splits["train"]
    for s in ["valid", "oot"]:
        tab = csi_table(train, ctx.splits[s], ctx.numeric, ctx.categorical)
        worst = tab.iloc[0]
        out.append(make(
            m, "stability", f"csi_max_train_to_{s}", worst["psi"],
            grade(worst["psi"], w, f), w, f, rule,
            {"worst_feature": worst["feature"],
             "n_features_above_warn": int((tab["psi"] > w).sum()),
             "n_features_above_fail": int((tab["psi"] > f).sum()),
             "top10": tab.head(10)}))
        for r in tab[tab["psi"] > w].itertuples():
            out.append(make(m, "stability", f"csi_{r.feature}_train_to_{s}", r.psi,
                            grade(r.psi, w, f), w, f, rule,
                            {"missing_expected": r.missing_expected,
                             "missing_actual": r.missing_actual}))

    dr_train = ctx.y("train").mean()
    for s in ["valid", "oot"]:
        out.append(make(m, "stability", f"default_rate_shift_train_to_{s}",
                        ctx.y(s).mean() / dr_train - 1, "info",
                        details={"train_dr": dr_train, f"{s}_dr": ctx.y(s).mean()}))
    return out
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/validation/out_of_time.py"] = r'''"""Out-of-time performance: does discrimination survive the move to a later period?

  - Gini / KS / AUC / Brier per split
  - relative Gini loss train -> oot and valid -> oot
  - Gini floor (minimum acceptable discriminatory power)
  - Gini by calendar quarter across 2014-2015 (stability inside the test window)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from finrisk.validation.discrimination import auc_gini_ks
from finrisk.validation.results import THRESHOLDS, grade, make


def performance_by_period(dates, y, p, freq: str = "Q", min_n: int = 500) -> pd.DataFrame:
    period = pd.to_datetime(pd.Series(np.asarray(dates))).dt.to_period(freq)
    df = pd.DataFrame({"period": period.to_numpy(), "y": np.asarray(y).astype(int),
                       "p": np.asarray(p, dtype=float)})
    rows = []
    for per, g in df.groupby("period"):
        if len(g) < min_n or g["y"].nunique() < 2:
            continue
        rows.append({"period": str(per), "n": len(g), "observed_dr": g["y"].mean(),
                     "mean_pd": g["p"].mean(), "gini": auc_gini_ks(g["y"], g["p"])["gini"]})
    return pd.DataFrame(rows)


def run(ctx) -> list:
    out, m = [], ctx.model_name
    met = {}
    for s in ["train", "valid", "oot"]:
        d = auc_gini_ks(ctx.y(s), ctx.pds[s])
        d["brier"] = float(brier_score_loss(ctx.y(s), np.clip(ctx.pds[s], 1e-6, 1 - 1e-6)))
        d["n"] = int(len(ctx.y(s)))
        met[s] = d
        out.append(make(m, "out_of_time", f"{s}_gini", d["gini"], "info", details=d))

    w, f = THRESHOLDS["gini_drop_relative"]
    for a, b in [("train", "oot"), ("valid", "oot")]:
        drop = (met[a]["gini"] - met[b]["gini"]) / met[a]["gini"]
        out.append(make(m, "out_of_time", f"gini_drop_{a}_to_{b}", drop, grade(drop, w, f), w, f,
                        f"relative Gini loss <= {w:.0%} pass, <= {f:.0%} warn, else fail",
                        {f"gini_{a}": met[a]["gini"], f"gini_{b}": met[b]["gini"]}))

    wf, ff = THRESHOLDS["gini_floor"]
    out.append(make(m, "out_of_time", "oot_gini_floor", met["oot"]["gini"],
                    grade(met["oot"]["gini"], wf, ff, higher_is_worse=False), wf, ff,
                    f"OOT Gini >= {wf} pass, >= {ff} warn, else fail"))

    both = pd.concat([ctx.splits["valid"], ctx.splits["oot"]], ignore_index=True)
    p_both = np.concatenate([ctx.pds["valid"], ctx.pds["oot"]])
    per = performance_by_period(both["issue_d"], both["is_default"], p_both)
    if len(per):
        worst = float(per["gini"].min())
        out.append(make(m, "out_of_time", "worst_quarter_gini", worst,
                        grade(worst, wf, ff, higher_is_worse=False), wf, ff,
                        "lowest quarterly Gini over 2014-2015, same floor as above",
                        {"by_quarter": per}))
    return out
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/validation/run_validation.py"] = r'''"""Run the validation suite on the champion and challenger and write the results.

Outputs (single source of truth for dashboard, PDF report and LLM):
    reports/validation_results.json
    results.validation_results   (PostgreSQL)

Usage:
    python -m finrisk.validation.run_validation
    python -m finrisk.validation.run_validation --feature-sets independent with_lender
    python -m finrisk.validation.run_validation --no-db
"""
from __future__ import annotations

import argparse
import importlib
from datetime import datetime

import pandas as pd

from finrisk.config import get_settings
from finrisk.db import connect
from finrisk.features.build import get_feature_columns, load_features
from finrisk.models import registry
from finrisk.models.registry import load_model_object
from finrisk.models.scorecard import Scorecard
from finrisk.validation import results as R

pd.set_option("display.width", 180)
pd.set_option("display.max_colwidth", 60)

# executed in this order; a suite without a run() function is skipped (not built yet)
SUITES = ["calibration", "out_of_time", "stability",
          "leakage", "explanation_stability", "fairness"]


def build_contexts(feature_sets: list[str]) -> list[R.ValidationContext]:
    client = registry.setup_mlflow()
    df = load_features()
    splits = {s: df[df["split"] == s].reset_index(drop=True) for s in R.SPLITS}
    contexts = []
    for fs in feature_sets:
        num, cat = get_feature_columns(fs)
        for role in ("champion", "challenger"):
            model = load_model_object(role, fs)
            kind = "scorecard" if isinstance(model, Scorecard) else model.kind
            version = client.get_model_version_by_alias(registry.registered_name(fs), role).version
            pds = {s: model.predict_proba(splits[s]) for s in R.SPLITS}
            contexts.append(R.ValidationContext(
                model_name=f"{kind}_{fs}", kind=kind, role=role, version=str(version),
                feature_set=fs, model=model, splits=splits, pds=pds,
                numeric=num, categorical=cat))
    return contexts


def run_suites(ctx: R.ValidationContext) -> list[R.TestResult]:
    results = []
    for name in SUITES:
        mod = importlib.import_module(f"finrisk.validation.{name}")
        fn = getattr(mod, "run", None)
        if fn is None:
            print(f"    - {name}: not implemented yet, skipped")
            continue
        res = fn(ctx)
        results.extend(res)
        print(f"    - {name}: {len(res)} results")
    return results


def summarize(results: list[R.TestResult]) -> None:
    df = pd.DataFrame([{"model": r.model, "test": r.test, "metric": r.metric,
                        "value": r.value, "status": r.status, "rule": r.rule} for r in results])
    counts = df.pivot_table(index="model", columns="status", values="metric",
                            aggfunc="count", fill_value=0)
    for col in ["pass", "warn", "fail", "info"]:
        if col not in counts:
            counts[col] = 0
    print("\n" + "=" * 100)
    print("VALIDATION SUMMARY (number of results by status)")
    print("=" * 100)
    print(counts[["pass", "warn", "fail", "info"]].to_string())

    flagged = df[df["status"].isin(["warn", "fail"])].copy()
    flagged["value"] = flagged["value"].map(lambda v: "n/a" if v is None else f"{v:.4f}")
    flagged["sev"] = flagged["status"].map({"fail": 0, "warn": 1})
    flagged = flagged.sort_values(["model", "sev", "test"]).drop(columns=["sev", "rule"])
    print("\n" + "=" * 100)
    print("FINDINGS (warn / fail)")
    print("=" * 100)
    print(flagged.to_string(index=False) if len(flagged) else "none")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-sets", nargs="+", default=["independent"])
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args()

    connect().close()  # friendly error if Docker is down
    print("Loading models and data ...")
    contexts = build_contexts(args.feature_sets)

    results: list[R.TestResult] = []
    for ctx in contexts:
        print(f"\n[{ctx.model_name}] role={ctx.role} version={ctx.version}")
        results.extend(run_suites(ctx))

    run_id = "val_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    models = {c.model_name: {"role": c.role, "version": c.version, "kind": c.kind,
                             "feature_set": c.feature_set} for c in contexts}
    path = get_settings().reports_dir / "validation_results.json"
    R.write_json(path, run_id, results, models)
    print(f"\nWrote {len(results)} results to {path}")
    if not args.no_db:
        R.write_db(run_id, results)
        print(f"Wrote results to PostgreSQL (results.validation_results, run_id={run_id})")

    summarize(results)


if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
FILES["tests/test_calibration.py"] = r'''"""Unit tests for calibration metrics and recalibration."""
import numpy as np
import pytest

from finrisk.validation.calibration import (
    Recalibrator, band_calibration, calibration_slope_intercept, calibration_table,
    hosmer_lemeshow, intercept_shift)


def _calibrated(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(2, 12, n)
    y = (rng.uniform(size=n) < p).astype(int)
    return y, p


def test_calibration_table_has_deciles_and_ratio_near_one_when_calibrated():
    y, p = _calibrated()
    t = calibration_table(y, p, 10)
    assert len(t) == 10
    assert t["n"].sum() == len(y)
    assert np.all(np.abs(t["ratio"] - 1) < 0.10)
    assert t["pd_inside_ci"].mean() >= 0.7


def test_hosmer_lemeshow_does_not_reject_a_calibrated_model():
    y, p = _calibrated(seed=1)
    assert hosmer_lemeshow(y, p)["p_value"] > 0.01


def test_hosmer_lemeshow_rejects_a_biased_model():
    y, p = _calibrated(seed=2)
    assert hosmer_lemeshow(y, p * 0.7)["p_value"] < 0.001


def test_slope_intercept_of_a_calibrated_model():
    y, p = _calibrated(seed=3)
    r = calibration_slope_intercept(y, p)
    assert r["slope"] == pytest.approx(1.0, abs=0.05)
    assert r["intercept"] == pytest.approx(0.0, abs=0.05)


def test_overconfident_predictions_have_slope_below_one():
    rng = np.random.default_rng(4)
    p_true = rng.beta(2, 12, 200_000)
    y = (rng.uniform(size=p_true.size) < p_true).astype(int)
    lg = np.log(p_true / (1 - p_true))
    p_over = 1 / (1 + np.exp(-2.0 * lg))            # twice as extreme as reality
    assert calibration_slope_intercept(y, p_over)["slope"] < 0.7


def test_intercept_recalibration_matches_observed_rate():
    y, p = _calibrated(seed=5)
    p_low = p * 0.75                                  # systematically too low
    rec = Recalibrator("intercept").fit(p_low, y)
    assert rec.transform(p_low).mean() == pytest.approx(y.mean(), abs=0.002)
    assert intercept_shift(y, p_low) > 0


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_other_recalibrations_reduce_bias(method):
    y, p = _calibrated(seed=6)
    p_low = p * 0.75
    rec = Recalibrator(method).fit(p_low, y)
    new = rec.transform(p_low)
    assert abs(new.mean() - y.mean()) < abs(p_low.mean() - y.mean())
    assert ((new > 0) & (new < 1)).all()


def test_band_calibration_flags_underprediction():
    rng = np.random.default_rng(7)
    p = rng.beta(2, 12, 200_000)
    y = (rng.uniform(size=p.size) < np.clip(p * 1.3, 0, 1)).astype(int)
    t = band_calibration(y, p)
    assert (t["p_value_under"] < 0.01).sum() >= 3
    assert (t["ratio"] > 1).all()
'''

# ---------------------------------------------------------------------------
FILES["tests/test_stability.py"] = r'''"""Unit tests for PSI / CSI."""
import numpy as np
import pandas as pd
import pytest

from finrisk.validation.stability import csi_table, psi_categorical, psi_label, psi_numeric


def test_identical_distributions_have_psi_near_zero():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=50_000), rng.normal(size=50_000)
    assert psi_numeric(a, b) < 0.005


def test_shifted_distribution_has_large_psi():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=50_000), rng.normal(loc=1.0, size=50_000)
    assert psi_numeric(a, b) > 0.25


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(2)
    a = rng.normal(size=50_000)
    small = psi_numeric(a, rng.normal(loc=0.2, size=50_000))
    big = psi_numeric(a, rng.normal(loc=0.8, size=50_000))
    assert 0 < small < big


def test_missing_values_are_a_separate_bin():
    rng = np.random.default_rng(3)
    a = rng.normal(size=50_000)
    b = a.copy()
    b[:20_000] = np.nan                               # 40% suddenly missing
    assert psi_numeric(a, b) > 0.25


def test_categorical_psi():
    a = np.array(["x"] * 700 + ["y"] * 300)
    assert psi_categorical(a, a) == pytest.approx(0.0, abs=1e-9)
    assert psi_categorical(a, np.array(["x"] * 300 + ["y"] * 700)) > 0.25
    assert psi_categorical(a, np.array(["z"] * 1000)) > 0.25     # unseen level


def test_csi_table_ranks_the_shifted_feature_first():
    rng = np.random.default_rng(4)
    n = 30_000
    e = pd.DataFrame({"stable": rng.normal(size=n), "shifted": rng.normal(size=n),
                      "cat": rng.choice(["a", "b"], n)})
    a = pd.DataFrame({"stable": rng.normal(size=n), "shifted": rng.normal(loc=1, size=n),
                      "cat": rng.choice(["a", "b"], n)})
    t = csi_table(e, a, ["stable", "shifted"], ["cat"])
    assert t.iloc[0]["feature"] == "shifted"
    assert t.set_index("feature").loc["stable", "label"] == "stable"


def test_psi_label_thresholds():
    assert psi_label(0.05) == "stable"
    assert psi_label(0.15) == "moderate shift"
    assert psi_label(0.40) == "significant shift"
'''

# ---------------------------------------------------------------------------
FILES["tests/test_validation_framework.py"] = r'''"""Tests for the results contract, grading and the out-of-time helpers."""
import json

import numpy as np
import pandas as pd

from finrisk.validation import results as R
from finrisk.validation.out_of_time import performance_by_period


def test_grade_higher_is_worse():
    assert R.grade(0.05, 0.10, 0.20) == "pass"
    assert R.grade(0.10, 0.10, 0.20) == "pass"        # boundary is inclusive
    assert R.grade(0.15, 0.10, 0.20) == "warn"
    assert R.grade(0.30, 0.10, 0.20) == "fail"


def test_grade_lower_is_worse():
    assert R.grade(0.40, 0.25, 0.15, higher_is_worse=False) == "pass"
    assert R.grade(0.20, 0.25, 0.15, higher_is_worse=False) == "warn"
    assert R.grade(0.10, 0.25, 0.15, higher_is_worse=False) == "fail"


def test_grade_uncomputable_value_is_warn():
    assert R.grade(float("nan"), 0.1, 0.2) == "warn"
    assert R.grade(None, 0.1, 0.2) == "warn"


def test_results_serialise_to_json_even_with_numpy_nan_and_dataframes():
    r = R.make("m", "t", "x", np.float32(0.5), "pass", 0.1, 0.2, "rule",
               {"arr": np.array([1, 2]), "nan": float("nan"), "n": np.int64(3),
                "df": pd.DataFrame({"a": [1.0, np.nan]}), "when": pd.Timestamp("2015-01-01")})
    d = r.to_dict()
    text = json.dumps(d)                              # must not raise
    back = json.loads(text)
    assert back["details"]["nan"] is None
    assert back["details"]["df"][1]["a"] is None
    assert back["details"]["arr"] == [1, 2]


def test_make_turns_nan_value_into_none():
    assert R.make("m", "t", "x", float("nan"), "warn").value is None


def test_thresholds_are_ordered_sensibly():
    for key, (warn, fail) in R.THRESHOLDS.items():
        if key == "gini_floor":
            assert fail < warn                        # lower is worse
        else:
            assert warn < fail


def test_performance_by_period_returns_quarterly_gini():
    rng = np.random.default_rng(0)
    n = 40_000
    dates = pd.date_range("2014-01-01", "2015-12-31", periods=n)
    p = rng.beta(2, 10, n)
    y = (rng.uniform(size=n) < p).astype(int)
    t = performance_by_period(dates, y, p)
    assert len(t) == 8
    assert (t["gini"] > 0.2).all()
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
        print(f"  warn    {rel}: expected text not found, skipping this patch")


def main() -> None:
    if not (ROOT / "src" / "finrisk" / "models" / "registry.py").exists():
        sys.exit("Run apply_phase2b.py first, from the repo root.")

    print(f"Applying Phase 3a in {ROOT}\n")
    for rel, content in FILES.items():
        write(rel, content)

    # cosmetic: hide negligible reasons ("+0.00 log-odds") from reason codes
    patch(
        "src/finrisk/explain/reason_codes.py",
        '''def top_reasons(contributions: pd.Series, values: pd.Series, k: int = 5) -> list[dict]:
    """Top-k features that INCREASE the applicant's risk (positive log-odds contribution)."""
    pos = contributions[contributions > 0].sort_values(ascending=False).head(k)''',
        '''def top_reasons(contributions: pd.Series, values: pd.Series, k: int = 5,
                min_impact: float = 0.01) -> list[dict]:
    """Top-k features that INCREASE the applicant's risk (log-odds contribution > min_impact)."""
    pos = contributions[contributions > min_impact].sort_values(ascending=False).head(k)''',
    )

    print("\nDone. Next:")
    print("  docker compose up -d")
    print("  pytest -q")
    print("  python -m finrisk.validation.run_validation")


if __name__ == "__main__":
    main()