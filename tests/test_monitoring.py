"""Tests for the monitoring metrics (no database needed)."""
import numpy as np
import pandas as pd
import pytest

from finrisk.monitoring.drift import (
    detection_summary, monthly_monitoring, recalibration_summary, to_month)

OUTCOMES_UNTIL = pd.Timestamp("2015-12-01")
LAG = 12


def _portfolio(n_per_month=2500, start="2014-01-01", months=36, seed=0,
               default_multiplier=1.0, score_shift_from=None, score_shift=1.0):
    rng = np.random.default_rng(seed)
    month_starts = pd.date_range(start, periods=months, freq="MS")
    frames = []
    for m in month_starts:
        n = n_per_month
        p_true = rng.beta(2, 12, n)
        shifted = score_shift_from is not None and m >= pd.Timestamp(score_shift_from)
        p = np.clip(p_true * (score_shift if shifted else 1.0), 1e-4, 0.99)
        y = (rng.uniform(size=n) < np.clip(p_true * default_multiplier, 0, 1)).astype(int)
        frames.append(pd.DataFrame({
            "issue_d": m + pd.to_timedelta(rng.integers(0, 27, n), unit="D"),
            "is_default": y, "p": p,
            "f1": rng.normal(size=n), "c1": rng.choice(["a", "b", "c"], n)}))
    df = pd.concat(frames, ignore_index=True)
    return df, df["p"].to_numpy()


def _reference(seed=99):
    df, p = _portfolio(n_per_month=2500, start="2012-01-01", months=24, seed=seed)
    return df, p


def _run(**kw):
    df, p = _portfolio(**{k: v for k, v in kw.items() if k in
                          ("default_multiplier", "score_shift_from", "score_shift", "seed")})
    ref, ref_p = _reference()
    return monthly_monitoring(df, p, ref, ref_p, ["f1"], ["c1"], OUTCOMES_UNTIL,
                              lag_months=kw.get("lag", LAG))


def test_to_month_truncates_to_month_start():
    out = to_month(["2015-03-17", "2015-03-01", "2016-12-31"])
    assert list(out) == [pd.Timestamp("2015-03-01"), pd.Timestamp("2015-03-01"),
                         pd.Timestamp("2016-12-01")]


def test_stable_portfolio_is_green_and_calibrated():
    t = _run()
    assert len(t) == 36
    assert t["score_psi"].max() < 0.10
    assert (t["status_score_psi"] == "green").all()
    ratios = t["calibration_ratio_roll3"].dropna()
    assert ((ratios - 1).abs() < 0.08).all()


def test_underprediction_is_detected_by_the_calibration_monitor():
    t = _run(default_multiplier=1.4)
    ratios = t["calibration_ratio_roll3"].dropna()
    assert (ratios > 1.25).all()
    assert (t["status_calibration"].dropna() == "red").all()
    # ...while the score distribution itself did not move, so score PSI stays green
    assert (t["status_score_psi"] == "green").all()


def test_months_without_known_outcomes_have_no_outcome_metrics():
    t = _run()
    late = t[t["month"] > OUTCOMES_UNTIL]
    assert len(late) == 12
    assert not late["outcomes_available"].any()
    for col in ["observed_default_rate", "calibration_ratio_roll3", "gini_roll3", "status_calibration"]:
        assert late[col].isna().all(), col
    assert late["score_psi"].notna().all()            # input monitors still work


def test_score_drift_raises_the_psi_monitor():
    t = _run(score_shift_from="2015-07-01", score_shift=1.8)
    before = t[t["month"] < pd.Timestamp("2015-07-01")]
    after = t[t["month"] >= pd.Timestamp("2015-07-01")]
    assert (before["status_score_psi"] == "green").all()
    assert (after["status_score_psi"] == "red").all()


def test_recalibration_uses_only_outcomes_known_at_the_time():
    t = _run(default_multiplier=1.3)
    with_window = t.dropna(subset=["recal_window_end"])
    assert len(with_window) > 0
    cutoff = with_window["month"] - pd.DateOffset(months=LAG)
    assert (with_window["recal_window_end"] <= cutoff).all()
    assert (with_window["recal_window_end"] <= OUTCOMES_UNTIL).all()
    early = t[t["month"] < pd.Timestamp("2014-01-01") + pd.DateOffset(months=LAG + 2)]
    assert early["recal_window_end"].isna().all()    # not enough known history yet


def test_recalibration_helps_when_the_bias_is_stable():
    t = _run(default_multiplier=1.3)
    rec = recalibration_summary(t)
    assert rec["months_compared"] > 0
    assert rec["mean_abs_deviation_recalibrated"] < rec["mean_abs_deviation_raw"]


def test_detection_summary_applies_the_outcome_lag_to_outcome_monitors_only():
    t = _run(default_multiplier=1.4, score_shift_from="2016-01-01", score_shift=2.0)
    det = detection_summary(t, lag_months=LAG)
    cal = det["calibration"]["first_amber_or_red"]
    first_vintage = pd.Timestamp(cal["vintage"] + "-01")
    assert pd.Timestamp(cal["visible_from"] + "-01") == first_vintage + pd.DateOffset(months=LAG)
    psi = det["score_psi"]["first_amber_or_red"]
    assert psi["vintage"] == psi["visible_from"]     # input monitors are visible immediately
    assert psi["vintage"].startswith("2016-01")


def test_a_monitor_that_never_fires_reports_none():
    t = _run()
    assert detection_summary(t)["score_psi"]["first_amber_or_red"] is None
