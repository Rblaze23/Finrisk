"""Leakage tests: does any input contain information from AFTER the loan was issued?

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
