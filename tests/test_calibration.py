"""Unit tests for calibration metrics and recalibration."""
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
