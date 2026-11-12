"""Vintage-availability sensitivity.

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
