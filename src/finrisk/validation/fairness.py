"""Fairness and proxy checks.

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
