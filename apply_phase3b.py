#!/usr/bin/env python3
"""
Phase 3b for FinRisk Lab: leakage, vintage sensitivity, explanation stability, fairness.

Run from the ROOT of your repo (after apply_phase3a.py):

    python apply_phase3b.py

Four files from Phase 3a are intentionally REPLACED by improved versions
(results.py, stability.py, run_validation.py, tests/test_validation_framework.py);
the old copies are saved next to them as *.bak.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(".").resolve()
FORCE = "--force" in sys.argv

FILES = {}
OVERWRITE = {
    "src/finrisk/validation/results.py",
    "src/finrisk/validation/stability.py",
    "src/finrisk/validation/run_validation.py",
    "tests/test_validation_framework.py",
}

# ---------------------------------------------------------------------------
FILES["src/finrisk/validation/results.py"] = r'''"""Validation results contract: one schema shared by the tests, dashboard, report and LLM.

Every test returns TestResult objects:
    {"model", "test", "metric", "value", "status", "warn", "fail", "rule", "details"}
status is one of: pass | warn | fail | info
Tests whose name ends with "_detail" are feature-level diagnostics: they are stored and
reported, but do not count in the headline totals.
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
    "vintage_gini_loss": (0.03, 0.10),       # Gini lost when vintage-sensitive features are removed
    "single_feature_gini": (0.40, 0.50),     # a single input this predictive is suspicious
    "explanation_rank_corr": (0.80, 0.60),   # rank correlation of feature importance (lower is worse)
    "fairness_group_dev": (0.10, 0.20),      # group calibration vs overall calibration
    "geo_importance_share": (0.02, 0.05),    # share of model importance on a geographic variable
}
LOWER_IS_WORSE = {"gini_floor", "explanation_rank_corr"}

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

    @property
    def is_detail(self) -> bool:
        return self.test.endswith("_detail")

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
FILES["src/finrisk/validation/stability.py"] = r'''"""Population Stability Index (PSI) and Characteristic Stability Index (CSI).

PSI = sum( (actual% - expected%) * ln(actual% / expected%) )  over bins.
  < 0.10  stable   |   0.10 - 0.25  moderate shift   |   > 0.25  significant shift

  Score PSI : shift of the model's PD distribution vs the training population
  CSI       : the same measure per input feature (missing values form their own bin)

Only features the model actually USES are graded; drifting features it does not use,
and every per-feature row, are diagnostics ("stability_detail") that do not inflate the
headline totals.
Note: when a bin disappears entirely (e.g. a field that was 16% missing in training and
never missing afterwards) the formula gives a very large PSI. The flag is real (a structural
break in data availability) but the exact magnitude is dominated by that empty bin.
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


def model_features(ctx) -> list[str]:
    """Features the model actually uses (scorecard: after selection; boosting: all)."""
    used = getattr(ctx.model, "features_", None)
    return list(used) if used else list(ctx.numeric) + list(ctx.categorical)


# ---------------------------------------------------------------------------
def run(ctx) -> list:
    out, m = [], ctx.model_name
    w, f = THRESHOLDS["psi"]
    rule = f"PSI <= {w} pass, <= {f} warn, else fail"

    for s in ["valid", "oot", "monitor"]:
        v = psi_numeric(ctx.pds["train"], ctx.pds[s])
        out.append(make(m, "stability", f"score_psi_train_to_{s}", v, grade(v, w, f), w, f, rule))

    used = set(model_features(ctx))
    train = ctx.splits["train"]
    for s in ["valid", "oot"]:
        tab = csi_table(train, ctx.splits[s], ctx.numeric, ctx.categorical)
        tab["used_by_model"] = tab["feature"].isin(used)
        tab_used = tab[tab["used_by_model"]]
        worst = tab_used.iloc[0]
        out.append(make(
            m, "stability", f"csi_max_used_feature_train_to_{s}", worst["psi"],
            grade(worst["psi"], w, f), w, f, rule + " (features used by the model only)",
            {"worst_feature": worst["feature"],
             "n_used_features": int(len(tab_used)),
             "n_used_above_warn": int((tab_used["psi"] > w).sum()),
             "n_used_above_fail": int((tab_used["psi"] > f).sum()),
             "top10_all_features": tab.head(10)}))
        for r in tab[tab["psi"] > w].itertuples():
            out.append(make(
                m, "stability_detail", f"csi_{r.feature}_train_to_{s}", r.psi,
                grade(r.psi, w, f) if r.used_by_model else "info", w, f,
                rule if r.used_by_model else "diagnostic: feature not used by this model",
                {"used_by_model": bool(r.used_by_model),
                 "missing_expected": r.missing_expected, "missing_actual": r.missing_actual}))

    dr_train = ctx.y("train").mean()
    for s in ["valid", "oot"]:
        out.append(make(m, "stability", f"default_rate_shift_train_to_{s}",
                        ctx.y(s).mean() / dr_train - 1, "info",
                        details={"train_dr": dr_train, f"{s}_dr": ctx.y(s).mean()}))
    return out
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/validation/sensitivity.py"] = r'''"""Vintage-availability sensitivity.

Some fields were mostly EMPTY for early loans and always populated afterwards (a change in
data collection, not in borrowers). Inside a model trained on 2012-2013 such a field's
"missing" bin acts as a vintage marker. Test: find those features, retrain WITHOUT them,
and compare out-of-time performance.

  - small Gini loss  -> the model does not depend on the artifact; dropping them is cheaper and safer
  - large Gini loss  -> performance leans on features with a structural break in availability
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finrisk.models.gbm import GBMModel
from finrisk.models.scorecard import Scorecard
from finrisk.validation.discrimination import auc_gini_ks
from finrisk.validation.results import THRESHOLDS, grade, make

MISSING_SPREAD = 0.10   # change in missing share between train years that marks a feature


def vintage_sensitive_features(train_df: pd.DataFrame, features: list[str],
                               threshold: float = MISSING_SPREAD) -> pd.DataFrame:
    """Features whose missing share differs by more than `threshold` between issue years."""
    miss = train_df[features].isna().groupby(train_df["issue_year"]).mean()
    spread = (miss.max() - miss.min()).rename("spread")
    out = pd.concat([spread, miss.T], axis=1)
    out.columns = [str(c) for c in out.columns]
    return out[out["spread"] > threshold].sort_values("spread", ascending=False)


def _refit(ctx, drop: list[str]):
    num = [f for f in ctx.numeric if f not in drop]
    cat = [f for f in ctx.categorical if f not in drop]
    tr, va = ctx.splits["train"], ctx.splits["valid"]
    if ctx.kind == "scorecard":
        old = ctx.model
        model = Scorecard(num, cat, min_iv=old.min_iv, max_corr=old.max_corr,
                          max_missing=old.max_missing, C=old.C,
                          min_bin_pct=old.min_bin_pct, max_bins=old.max_bins)
        model.fit(tr, tr["is_default"])
    else:
        model = GBMModel(ctx.kind, num, cat, monotone=ctx.model.monotone, params=ctx.model.params)
        model.fit(tr, tr["is_default"], va, va["is_default"])
    return model


def run(ctx) -> list:
    out, m = [], ctx.model_name
    all_features = list(ctx.numeric) + list(ctx.categorical)
    sens = vintage_sensitive_features(ctx.splits["train"], all_features)
    out.append(make(m, "sensitivity", "vintage_sensitive_features", len(sens), "info",
                    details={"features": list(sens.index), "missing_share_by_year": sens}))
    if sens.empty:
        return out

    drop = list(sens.index)
    reduced = _refit(ctx, drop)
    p = reduced.predict_proba(ctx.splits["oot"])
    y = ctx.y("oot")
    g_new, g_old = auc_gini_ks(y, p), auc_gini_ks(y, ctx.pds["oot"])
    loss = (g_old["gini"] - g_new["gini"]) / g_old["gini"]
    w, f = THRESHOLDS["vintage_gini_loss"]
    used_now = set(getattr(ctx.model, "features_", all_features))

    out.append(make(m, "sensitivity", "oot_gini_without_vintage_features", g_new["gini"], "info",
                    details={"removed": drop, "gini_original": g_old["gini"],
                             "ks_original": g_old["ks"], "ks_reduced": g_new["ks"],
                             "calibration_ratio_original": float(y.mean() / ctx.pds["oot"].mean()),
                             "calibration_ratio_reduced": float(y.mean() / p.mean()),
                             "removed_features_used_by_original": sorted(used_now & set(drop))}))
    out.append(make(m, "sensitivity", "vintage_gini_loss", loss, grade(loss, w, f), w, f,
                    f"relative OOT Gini loss when vintage-sensitive features are removed: "
                    f"<= {w:.0%} pass, <= {f:.0%} warn, else fail",
                    {"gini_original": g_old["gini"], "gini_reduced": g_new["gini"]}))
    return out
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/validation/leakage.py"] = r'''"""Leakage tests: does any input contain information from AFTER the loan was issued?

  1. forbidden-feature check : no model input may be a post-origination column (hard rule)
  2. too-good-to-be-true     : no single input should predict default almost on its own
  3. demonstration           : retrain WITH the post-origination columns to show what leakage
                               looks like (a near-perfect Gini that would never exist in production)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sqlalchemy import text

from finrisk.data.columns import POST_ORIGINATION_COLUMNS
from finrisk.db import get_engine
from finrisk.models.gbm import GBMModel
from finrisk.validation.discrimination import auc_gini_ks
from finrisk.validation.results import THRESHOLDS, grade, make

LEAK_NUMERIC = ["total_pymnt", "total_rec_prncp", "recoveries", "last_pymnt_amnt",
                "out_prncp", "total_rec_late_fee", "collection_recovery_fee",
                "last_fico_range_high", "last_fico_range_low"]

_cache: dict = {}


def forbidden_overlap(model_features: list[str]) -> list[str]:
    return sorted(set(model_features) & set(POST_ORIGINATION_COLUMNS))


def single_feature_gini(df: pd.DataFrame, columns: list[str], y) -> pd.DataFrame:
    """|Gini| of each column used alone as a score (missing values -> median)."""
    y = np.asarray(y).astype(int)
    rows = []
    for c in columns:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() == 0 or x.nunique() < 2:
            rows.append({"feature": c, "gini": 0.0})
            continue
        auc = roc_auc_score(y, x.fillna(x.median()))
        rows.append({"feature": c, "gini": float(abs(2 * auc - 1))})
    return pd.DataFrame(rows).sort_values("gini", ascending=False).reset_index(drop=True)


def load_leak_columns() -> pd.DataFrame:
    if "leak" not in _cache:
        cols = ", ".join(f"c.{c}" for c in LEAK_NUMERIC)
        q = text(f"""SELECT c.loan_id, {cols}
                     FROM staging.loans_clean c
                     JOIN staging.loan_splits s USING (loan_id)
                     WHERE s.split IN ('train', 'valid', 'oot')""")
        df = pd.read_sql(q, get_engine())
        for c in LEAK_NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        _cache["leak"] = df
    return _cache["leak"]


def leaky_demo(ctx) -> dict:
    """Retrain a quick LightGBM that is allowed to see post-origination columns."""
    key = ("demo", ctx.feature_set)
    if key in _cache:
        return _cache[key]
    leak = load_leak_columns()
    tr, va, oo = (ctx.splits[s].merge(leak, on="loan_id", how="left")
                  for s in ["train", "valid", "oot"])
    model = GBMModel("lightgbm", ctx.numeric + LEAK_NUMERIC, ctx.categorical, monotone=False,
                     params=dict(n_estimators=300, learning_rate=0.1, num_leaves=15))
    model.fit(tr, tr["is_default"], va, va["is_default"])
    metrics = auc_gini_ks(oo["is_default"], model.predict_proba(oo))
    metrics["single_feature_gini"] = single_feature_gini(oo, LEAK_NUMERIC, oo["is_default"])
    _cache[key] = metrics
    return metrics


def run(ctx) -> list:
    out, m = [], ctx.model_name
    features = list(ctx.numeric) + list(ctx.categorical)

    bad = forbidden_overlap(features)
    out.append(make(m, "leakage", "post_origination_features_in_model", len(bad),
                    "pass" if not bad else "fail", 0, 0,
                    "no model input may be a post-origination column",
                    {"forbidden_columns_checked": POST_ORIGINATION_COLUMNS, "found": bad}))

    oot = ctx.splits["oot"]
    screen = single_feature_gini(oot, ctx.numeric, ctx.y("oot"))
    w, f = THRESHOLDS["single_feature_gini"]
    top = screen.iloc[0]
    out.append(make(m, "leakage", "max_single_feature_gini", top["gini"], grade(top["gini"], w, f), w, f,
                    f"strongest single input Gini <= {w} pass, <= {f} warn, else fail "
                    "(a single input this predictive usually means leakage)",
                    {"strongest": top["feature"], "top5": screen.head(5)}))

    demo = leaky_demo(ctx)
    clean_gini = auc_gini_ks(ctx.y("oot"), ctx.pds["oot"])["gini"]
    out.append(make(m, "leakage", "demonstration_leaky_model_oot_gini", demo["gini"], "info",
                    details={"clean_model_gini": clean_gini,
                             "inflation": demo["gini"] - clean_gini,
                             "leaky_ks": demo["ks"],
                             "single_feature_gini_of_post_origination_columns": demo["single_feature_gini"],
                             "note": "retrained with post-origination columns to show the effect; "
                                     "the production model excludes them"}))
    return out
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/validation/explanation_stability.py"] = r'''"""Explanation stability: does the model rely on the same drivers over time and across models?

Global importance (mean |contribution|) is computed on samples of train (2012-13), valid (2014)
and OOT (2015). We compare the rankings with Spearman correlation and top-5 overlap.
For the challenger we also compare its drivers with the champion's.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from finrisk.explain.shap_explainer import PDExplainer
from finrisk.validation.results import THRESHOLDS, grade, make

SAMPLE = 20_000


def rank_corr(a: pd.Series, b: pd.Series, min_share: float = 0.01) -> float:
    """Spearman correlation of two importance vectors over the features that matter in either."""
    idx = a.index.union(b.index)
    a, b = a.reindex(idx).fillna(0.0), b.reindex(idx).fillna(0.0)
    keep = (a / a.sum() >= min_share) | (b / b.sum() >= min_share)
    if keep.sum() < 3:
        return float("nan")
    return float(spearmanr(a[keep], b[keep]).correlation)


def top_k_overlap(a: pd.Series, b: pd.Series, k: int = 5) -> float:
    ta, tb = set(a.nlargest(k).index), set(b.nlargest(k).index)
    return len(ta & tb) / k


def importance_share(imp: pd.Series) -> pd.Series:
    return imp / imp.sum()


def run(ctx) -> list:
    out, m = [], ctx.model_name
    w, f = THRESHOLDS["explanation_rank_corr"]
    rule = f"Spearman rank correlation >= {w} pass, >= {f} warn, else fail"

    bg = ctx.splits["train"].sample(min(5000, len(ctx.splits["train"])), random_state=0)
    ex = PDExplainer(ctx.model, bg)
    samples = {s: ctx.splits[s].sample(min(SAMPLE, len(ctx.splits[s])), random_state=0)
               for s in ["train", "valid", "oot"]}
    imp = {s: ex.global_importance(df) for s, df in samples.items()}

    for s in ["valid", "oot"]:
        rho = rank_corr(imp["train"], imp[s])
        out.append(make(m, "explanation_stability", f"importance_rank_corr_train_to_{s}", rho,
                        grade(rho, w, f, higher_is_worse=False), w, f, rule,
                        {"top5_overlap": top_k_overlap(imp["train"], imp[s])}))

    shares = pd.DataFrame({s: importance_share(v) for s, v in imp.items()}).fillna(0.0)
    shares["change_train_to_oot"] = shares["oot"] - shares["train"]
    biggest = shares["change_train_to_oot"].abs().sort_values(ascending=False).index[0]
    out.append(make(m, "explanation_stability", "largest_importance_share_change_train_to_oot",
                    abs(shares.loc[biggest, "change_train_to_oot"]), "info",
                    details={"feature": biggest, "importance_shares": shares.sort_values("oot", ascending=False).head(12)}))

    if ctx.role == "challenger":
        from finrisk.models.registry import load_model_object
        champ = load_model_object("champion", ctx.feature_set)
        champ_imp = PDExplainer(champ, bg).global_importance(samples["oot"])
        rho = rank_corr(champ_imp, imp["oot"])
        out.append(make(m, "explanation_stability", "importance_rank_corr_champion_vs_challenger_oot",
                        rho, grade(rho, w, f, higher_is_worse=False), w, f,
                        rule + " (do the two models rely on the same drivers?)",
                        {"top5_overlap": top_k_overlap(champ_imp, imp["oot"]),
                         "champion_top5": list(champ_imp.nlargest(5).index),
                         "challenger_top5": list(imp["oot"].nlargest(5).index)}))
    return out
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/validation/fairness.py"] = r'''"""Fairness and proxy checks.

IMPORTANT LIMITATION: the LendingClub data contains no race, gender or age. These tests are
therefore PROXY analyses, not a legal fair-lending assessment:

  1. Group calibration : is the model equally calibrated across groups (state, income, home
                         ownership, verification, employment length)? A group whose
                         observed/predicted ratio differs a lot from the portfolio's is
                         systematically over- or under-scored.
  2. Geographic proxy  : how much of the model's importance sits on state of residence, which
                         can stand in for protected characteristics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finrisk.explain.shap_explainer import PDExplainer
from finrisk.models.risk_bands import assign_risk_band
from finrisk.validation.discrimination import auc_gini_ks
from finrisk.validation.results import THRESHOLDS, grade, make

MIN_GROUP_N = 2000
GEO_FEATURES = ["addr_state", "zip3"]


def group_calibration_table(group: pd.Series, y, p, overall_ratio: float,
                            min_n: int = MIN_GROUP_N) -> pd.DataFrame:
    df = pd.DataFrame({"g": group.astype(str).to_numpy(), "y": np.asarray(y).astype(int),
                       "p": np.asarray(p, dtype=float)})
    df["high_risk"] = np.isin(assign_risk_band(df["p"]), ["R5", "R6"])
    rows = []
    for g, d in df.groupby("g"):
        row = {"group": g, "n": len(d), "mean_pd": d["p"].mean(), "observed_dr": d["y"].mean(),
               "share_high_risk_bands": d["high_risk"].mean()}
        row["ratio"] = row["observed_dr"] / row["mean_pd"]
        row["ratio_vs_overall"] = row["ratio"] / overall_ratio - 1
        row["gini"] = auc_gini_ks(d["y"], d["p"])["gini"] if d["y"].nunique() > 1 and len(d) >= min_n else np.nan
        row["graded"] = len(d) >= min_n
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ratio_vs_overall", key=abs, ascending=False).reset_index(drop=True)


def define_groups(df: pd.DataFrame) -> dict[str, pd.Series]:
    g = {}
    top = df["addr_state"].value_counts().head(10).index
    g["addr_state"] = df["addr_state"].where(df["addr_state"].isin(top), "OTHER")
    q = pd.qcut(df["annual_inc"], 4, labels=False, duplicates="drop")
    g["income_quartile"] = q.map(lambda v: "unknown" if pd.isna(v) else f"Q{int(v) + 1}")
    g["home_ownership"] = df["home_ownership"].where(df["home_ownership"].isin(["RENT", "MORTGAGE", "OWN"]), "OTHER")
    g["verification_status"] = df["verification_status"].fillna("unknown")
    e = df["emp_length_years"]
    bucket = pd.cut(e, [-1, 0.5, 4.5, 9.5, 99], labels=["<1y", "1-4y", "5-9y", "10y+"]).astype(object)
    g["emp_length"] = bucket.where(e.notna(), "unknown")
    return g


def run(ctx) -> list:
    out, m = [], ctx.model_name
    oot, y, p = ctx.splits["oot"], ctx.y("oot"), ctx.pds["oot"]
    overall = float(y.mean() / p.mean())
    w, f = THRESHOLDS["fairness_group_dev"]
    rule = (f"largest group deviation of calibration ratio from the overall ratio "
            f"(groups with n >= {MIN_GROUP_N}): <= {w:.0%} pass, <= {f:.0%} warn, else fail")

    for attr, series in define_groups(oot).items():
        tab = group_calibration_table(series, y, p, overall)
        graded = tab[tab["graded"]]
        dev = float(graded["ratio_vs_overall"].abs().max()) if len(graded) else float("nan")
        worst = graded.iloc[0]["group"] if len(graded) else None
        out.append(make(m, "fairness", f"oot_group_calibration_dev_{attr}", dev, grade(dev, w, f), w, f, rule,
                        {"overall_ratio": overall, "worst_group": worst, "groups": tab,
                         "limitation": "proxy analysis: no protected attributes in the data"}))

    bg = ctx.splits["train"].sample(min(5000, len(ctx.splits["train"])), random_state=0)
    imp = PDExplainer(ctx.model, bg).global_importance(
        oot.sample(min(20_000, len(oot)), random_state=0))
    share = imp / imp.sum()
    geo = float(sum(share.get(c, 0.0) for c in GEO_FEATURES))
    wg, fg = THRESHOLDS["geo_importance_share"]
    out.append(make(m, "fairness", "geographic_importance_share", geo, grade(geo, wg, fg), wg, fg,
                    f"share of model importance on geographic variables {GEO_FEATURES}: "
                    f"<= {wg:.0%} pass, <= {fg:.0%} warn, else fail (potential proxy; needs legal review)",
                    {"importance_share": share.sort_values(ascending=False).head(10)}))
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
import time
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

# executed in this order; a suite without a run() function is skipped
SUITES = ["calibration", "out_of_time", "stability", "sensitivity",
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
        t0 = time.time()
        res = fn(ctx)
        results.extend(res)
        print(f"    - {name}: {len(res)} results ({time.time() - t0:.0f}s)")
    return results


def summarize(results: list[R.TestResult]) -> None:
    df = pd.DataFrame([{"model": r.model, "test": r.test, "metric": r.metric,
                        "value": r.value, "status": r.status, "detail": r.is_detail}
                       for r in results])
    head = df[~df["detail"]]
    counts = head.pivot_table(index="model", columns="status", values="metric",
                              aggfunc="count", fill_value=0)
    for col in ["pass", "warn", "fail", "info"]:
        if col not in counts:
            counts[col] = 0
    print("\n" + "=" * 100)
    print("VALIDATION SUMMARY (headline results by status; feature-level diagnostics excluded)")
    print("=" * 100)
    print(counts[["pass", "warn", "fail", "info"]].to_string())

    def table(frame):
        t = frame[frame["status"].isin(["warn", "fail"])].copy()
        t["value"] = t["value"].map(lambda v: "n/a" if v is None else f"{v:.4f}")
        t["sev"] = t["status"].map({"fail": 0, "warn": 1})
        return t.sort_values(["model", "sev", "test"])[["model", "test", "metric", "value", "status"]]

    flagged = table(head)
    print("\n" + "=" * 100)
    print("FINDINGS (warn / fail)")
    print("=" * 100)
    print(flagged.to_string(index=False) if len(flagged) else "none")

    diag = table(df[df["detail"]])
    if len(diag):
        print("\n" + "=" * 100)
        print(f"FEATURE-LEVEL DIAGNOSTICS flagged on features the model uses: {len(diag)} rows "
              "(full list in the JSON)")
        print("=" * 100)
        print(diag.head(15).to_string(index=False))


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
        if key in R.LOWER_IS_WORSE:
            assert fail < warn, key                   # lower is worse
        else:
            assert warn < fail, key


def test_detail_tests_are_flagged_as_diagnostics():
    assert R.make("m", "stability_detail", "x", 1.0, "warn").__class__ is R.TestResult
    r = R.TestResult("m", "stability_detail", "x", 1.0, "warn")
    assert r.is_detail
    assert not R.TestResult("m", "stability", "x", 1.0, "warn").is_detail


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
FILES["tests/test_validation_suites.py"] = r'''"""Tests for the leakage, sensitivity, explanation-stability and fairness helpers."""
import numpy as np
import pandas as pd
import pytest

from finrisk.validation.explanation_stability import rank_corr, top_k_overlap
from finrisk.validation.fairness import define_groups, group_calibration_table
from finrisk.validation.leakage import forbidden_overlap, single_feature_gini
from finrisk.validation.sensitivity import vintage_sensitive_features


# ---- leakage ---------------------------------------------------------------
def test_single_feature_gini_detects_a_leaky_column_and_ignores_noise():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20_000)
    df = pd.DataFrame({"leak": y + rng.normal(0, 0.05, y.size),
                       "inverse_leak": -y + rng.normal(0, 0.05, y.size),
                       "noise": rng.normal(size=y.size)})
    t = single_feature_gini(df, ["leak", "inverse_leak", "noise"], y).set_index("feature")
    assert t.loc["leak", "gini"] > 0.95
    assert t.loc["inverse_leak", "gini"] > 0.95       # direction does not matter
    assert t.loc["noise", "gini"] < 0.05


def test_single_feature_gini_handles_constant_and_all_missing_columns():
    y = np.array([0, 1] * 100)
    df = pd.DataFrame({"const": 1.0, "empty": np.nan}, index=range(len(y)))
    t = single_feature_gini(df, ["const", "empty"], y)
    assert (t["gini"] == 0).all()


def test_forbidden_overlap_finds_post_origination_columns():
    assert forbidden_overlap(["fico_mid", "total_pymnt", "recoveries"]) == ["recoveries", "total_pymnt"]
    assert forbidden_overlap(["fico_mid", "dti"]) == []


# ---- sensitivity -----------------------------------------------------------
def test_vintage_sensitive_features_flags_availability_breaks_only():
    rng = np.random.default_rng(1)
    n = 6000
    year = np.repeat([2012, 2013], n // 2)
    a = rng.normal(size=n)
    a[(year == 2012) & (rng.uniform(size=n) < 0.5)] = np.nan      # 50% missing in 2012 only
    b = rng.normal(size=n)
    b[rng.uniform(size=n) < 0.05] = np.nan                        # steady 5% missing
    df = pd.DataFrame({"issue_year": year, "a": a, "b": b})
    out = vintage_sensitive_features(df, ["a", "b"])
    assert list(out.index) == ["a"]
    assert out.loc["a", "spread"] > 0.4


# ---- explanation stability ---------------------------------------------------
def test_rank_corr_identical_and_reversed():
    a = pd.Series({"f1": 0.4, "f2": 0.3, "f3": 0.2, "f4": 0.1})
    assert rank_corr(a, a) == pytest.approx(1.0)
    assert rank_corr(a, a[::-1].set_axis(a.index)) == pytest.approx(-1.0)


def test_top_k_overlap():
    a = pd.Series({"f1": 5, "f2": 4, "f3": 3, "f4": 2, "f5": 1, "f6": 0.5})
    b = pd.Series({"f1": 5, "f2": 4, "f3": 3, "f4": 0.1, "f5": 0.2, "f6": 2})
    assert top_k_overlap(a, a, k=3) == 1.0
    assert top_k_overlap(a, b, k=4) == pytest.approx(0.75)


# ---- fairness ----------------------------------------------------------------
def test_group_calibration_flags_a_systematically_under_scored_group():
    rng = np.random.default_rng(2)
    n = 40_000
    g = pd.Series(rng.choice(["A", "B"], n))
    p_true = rng.beta(2, 12, n)
    y = (rng.uniform(size=n) < p_true).astype(int)
    p = np.where(g == "B", p_true * 0.6, p_true)      # model under-scores group B
    overall = y.mean() / p.mean()
    t = group_calibration_table(g, y, p, overall).set_index("group")
    assert abs(t.loc["B", "ratio_vs_overall"]) > 0.15
    assert abs(t.loc["A", "ratio_vs_overall"]) < abs(t.loc["B", "ratio_vs_overall"])


def test_small_groups_are_not_graded():
    g = pd.Series(["A"] * 5000 + ["tiny"] * 50)
    y = np.zeros(5050, dtype=int)
    y[::7] = 1
    p = np.full(5050, 0.14)
    t = group_calibration_table(g, y, p, 1.0).set_index("group")
    assert bool(t.loc["A", "graded"]) and not bool(t.loc["tiny", "graded"])


def test_define_groups_buckets_employment_and_income():
    df = pd.DataFrame({
        "addr_state": ["CA"] * 6 + ["NY"] * 2, "annual_inc": np.arange(1, 9) * 10_000.0,
        "home_ownership": ["RENT", "OWN", "MORTGAGE", "NONE", "RENT", "OWN", "RENT", "OWN"],
        "verification_status": ["Verified"] * 7 + [None],
        "emp_length_years": [0, 1, 4, 5, 9, 10, np.nan, 2]})
    g = define_groups(df)
    assert set(g) == {"addr_state", "income_quartile", "home_ownership",
                      "verification_status", "emp_length"}
    assert g["emp_length"].iloc[0] == "<1y" and g["emp_length"].iloc[6] == "unknown"
    assert g["home_ownership"].iloc[3] == "OTHER"
    assert g["income_quartile"].nunique() == 4
'''


# ---------------------------------------------------------------------------
def write(rel: str, content: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if rel in OVERWRITE:
            if old == content:
                print(f"  same    {rel}")
                return
            shutil.copy(path, str(path) + ".bak")
            path.write_text(content, encoding="utf-8")
            print(f"  replace {rel}  (old copy saved as .bak)")
            return
        if not FORCE and len(old.splitlines()) > 3:
            print(
                f"  skip    {rel}  (already edited, use --force to overwrite)")
            return
    path.write_text(content, encoding="utf-8")
    print(f"  write   {rel}")


def main() -> None:
    if not (ROOT / "src" / "finrisk" / "validation" / "calibration.py").exists() or \
       "def run" not in (ROOT / "src" / "finrisk" / "validation" / "calibration.py").read_text(encoding="utf-8"):
        sys.exit("Run apply_phase3a.py first, from the repo root.")

    print(f"Applying Phase 3b in {ROOT}\n")
    for rel, content in FILES.items():
        write(rel, content)

    print("\nDone. Next:")
    print("  docker compose up -d")
    print("  pytest -q")
    print("  python -m finrisk.validation.run_validation")


if __name__ == "__main__":
    main()
