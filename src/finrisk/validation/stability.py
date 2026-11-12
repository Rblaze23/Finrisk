"""Population Stability Index (PSI) and Characteristic Stability Index (CSI).

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
