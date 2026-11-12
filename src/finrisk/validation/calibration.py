"""Calibration: do predicted PDs match observed default rates?

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
