"""Explanation stability: does the model rely on the same drivers over time and across models?

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
