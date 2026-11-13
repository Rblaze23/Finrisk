"""Portfolio monitoring metrics, computed per issue-month ("vintage").

For every month the bank would track:
  - volume, mean predicted PD, share of loans in the two riskiest bands
  - score PSI (PD distribution vs the training population)
  - CSI of every input the model uses; the worst one is reported
  - observed default rate, calibration ratio and Gini (pooled over 3 months) WHEN outcomes are known
  - a lagged recalibration: an intercept shift fitted only on outcomes that would already be
    observable at that time (vintage <= month - outcome_lag), applied to the new month

Outcomes are not instantly known in real life. `outcome_lag_months` models that delay, and
vintages after `outcomes_until` have no reliable outcomes at all (censored in this dataset).

Traffic lights reuse the validation thresholds: green = pass, amber = warn, red = fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finrisk.models.risk_bands import assign_risk_band
from finrisk.validation.calibration import Recalibrator
from finrisk.validation.discrimination import auc_gini_ks
from finrisk.validation.results import THRESHOLDS, grade
from finrisk.validation.stability import psi_categorical, psi_numeric

LIGHT = {"pass": "green", "warn": "amber", "fail": "red"}
ALERT_MONITORS = {   # monitor -> (status column, needs known outcomes)
    "score_psi": ("status_score_psi", False),
    "csi": ("status_csi", False),
    "calibration": ("status_calibration", True),
    "gini": ("status_gini", True),
}


def to_month(dates) -> pd.Series:
    """Calendar month start for each date."""
    return pd.to_datetime(pd.Series(np.asarray(dates))).dt.to_period("M").dt.to_timestamp()


def _light(value, warn, fail, higher_is_worse=True):
    if value is None or not np.isfinite(value):
        return None
    return LIGHT[grade(value, warn, fail, higher_is_worse)]


def _rolling_ratio(defaults: pd.Series, expected: pd.Series, window: int = 3) -> pd.Series:
    roll = lambda s: s.rolling(window, min_periods=window).sum()
    return roll(defaults) / roll(expected)


def monthly_monitoring(df: pd.DataFrame, p, ref_df: pd.DataFrame, ref_p,
                       numeric: list[str], categorical: list[str],
                       outcomes_until, lag_months: int = 12,
                       recal_window: int = 6, min_known: int = 3) -> pd.DataFrame:
    d = df.reset_index(drop=True)[["issue_d", "is_default"] + list(numeric) + list(categorical)].copy()
    d["p"] = np.asarray(p, dtype=float)
    d["month"] = to_month(d["issue_d"]).to_numpy()
    outcomes_until = pd.Timestamp(outcomes_until)
    groups = {m: g for m, g in d.groupby("month")}
    months = sorted(groups)

    wp, fp = THRESHOLDS["psi"]
    wc, fc = THRESHOLDS["calibration_ratio_dev"]
    wg, fg = THRESHOLDS["gini_floor"]

    rows = []
    for m in months:
        g = groups[m]
        has_out = m <= outcomes_until
        csi = {f: psi_numeric(ref_df[f], g[f]) for f in numeric}
        csi.update({f: psi_categorical(ref_df[f], g[f]) for f in categorical})
        worst = max(csi, key=lambda k: -1.0 if not np.isfinite(csi[k]) else csi[k])

        row = {
            "month": m, "n_loans": len(g), "mean_pd": float(g["p"].mean()),
            "share_high_risk": float(np.isin(assign_risk_band(g["p"]), ["R5", "R6"]).mean()),
            "score_psi": psi_numeric(ref_p, g["p"]),
            "csi_max_used": float(csi[worst]), "csi_worst_feature": worst,
            "outcomes_available": bool(has_out),
            "n_defaults": float(g["is_default"].sum()) if has_out else np.nan,
            "observed_default_rate": float(g["is_default"].mean()) if has_out else np.nan,
            "gini": auc_gini_ks(g["is_default"], g["p"])["gini"]
                    if has_out and g["is_default"].nunique() > 1 else np.nan,
        }

        # lagged recalibration: use only outcomes already observable at month m
        known_cut = m - pd.DateOffset(months=lag_months)
        known = [k for k in months if k <= known_cut and k <= outcomes_until]
        window = known[-recal_window:]
        row.update({"recal_window_start": pd.NaT, "recal_window_end": pd.NaT,
                    "recal_delta": np.nan, "mean_pd_recal": np.nan})
        if len(window) >= min_known:
            w = d[d["month"].isin(window)]
            rec = Recalibrator("intercept").fit(w["p"], w["is_default"])
            row.update({"recal_window_start": window[0], "recal_window_end": window[-1],
                        "recal_delta": float(rec.delta_),
                        "mean_pd_recal": float(rec.transform(g["p"]).mean())})
        rows.append(row)

    t = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)

    exp = t["mean_pd"] * t["n_loans"]
    exp_recal = t["mean_pd_recal"] * t["n_loans"]
    t["calibration_ratio_roll3"] = _rolling_ratio(t["n_defaults"], exp)
    t["calibration_ratio_recal_roll3"] = _rolling_ratio(t["n_defaults"], exp_recal)

    gini3 = []
    for i in range(len(t)):
        win = t.iloc[max(0, i - 2): i + 1]
        if len(win) < 3 or not win["outcomes_available"].all():
            gini3.append(np.nan)
            continue
        sub = d[d["month"].isin(win["month"])]
        gini3.append(auc_gini_ks(sub["is_default"], sub["p"])["gini"])
    t["gini_roll3"] = gini3

    t["status_score_psi"] = [_light(v, wp, fp) for v in t["score_psi"]]
    t["status_csi"] = [_light(v, wp, fp) for v in t["csi_max_used"]]
    t["status_calibration"] = [_light(abs(v - 1) if np.isfinite(v) else None, wc, fc)
                               for v in t["calibration_ratio_roll3"]]
    t["status_calibration_recal"] = [_light(abs(v - 1) if np.isfinite(v) else None, wc, fc)
                                     for v in t["calibration_ratio_recal_roll3"]]
    t["status_gini"] = [_light(v, wg, fg, higher_is_worse=False) for v in t["gini_roll3"]]
    return t


def detection_summary(t: pd.DataFrame, lag_months: int = 12) -> dict:
    """When would each monitor have first raised an amber / red alert, and when would it be VISIBLE?

    Input monitors (score PSI, CSI) are visible in the month itself; outcome monitors only
    `lag_months` later, once the outcomes of that vintage are observable.
    """
    out = {}
    for name, (col, needs_outcomes) in ALERT_MONITORS.items():
        s = t[["month", col]].dropna()
        lag = lag_months if needs_outcomes else 0

        def first(levels):
            hit = s[s[col].isin(levels)]
            if hit.empty:
                return None
            m = hit["month"].min()
            return {"vintage": m.strftime("%Y-%m"),
                    "visible_from": (m + pd.DateOffset(months=lag)).strftime("%Y-%m")}

        out[name] = {"first_amber_or_red": first(["amber", "red"]), "first_red": first(["red"]),
                     "months_evaluated": int(len(s)),
                     "months_red": int((s[col] == "red").sum()),
                     "months_amber": int((s[col] == "amber").sum())}
    return out


def recalibration_summary(t: pd.DataFrame) -> dict:
    """Does a lagged recalibration keep the calibration ratio closer to 1 than doing nothing?"""
    both = t.dropna(subset=["calibration_ratio_roll3", "calibration_ratio_recal_roll3"])
    if both.empty:
        return {"months_compared": 0}
    return {
        "months_compared": int(len(both)),
        "mean_abs_deviation_raw": float((both["calibration_ratio_roll3"] - 1).abs().mean()),
        "mean_abs_deviation_recalibrated": float((both["calibration_ratio_recal_roll3"] - 1).abs().mean()),
        "last_ratio_raw": float(both["calibration_ratio_roll3"].iloc[-1]),
        "last_ratio_recalibrated": float(both["calibration_ratio_recal_roll3"].iloc[-1]),
        "last_month": both["month"].iloc[-1].strftime("%Y-%m"),
    }
