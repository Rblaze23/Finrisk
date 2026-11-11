"""Unit tests for AUC / Gini / KS."""
import numpy as np
import pytest

from finrisk.validation.discrimination import auc_gini_ks


def test_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = auc_gini_ks(y, s)
    assert m["auc"] == pytest.approx(1.0)
    assert m["gini"] == pytest.approx(1.0)
    assert m["ks"] == pytest.approx(1.0)


def test_random_score_has_gini_near_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20000)
    s = rng.uniform(size=20000)
    assert abs(auc_gini_ks(y, s)["gini"]) < 0.03


def test_inverted_score_has_negative_gini():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    assert auc_gini_ks(y, s)["gini"] == pytest.approx(-1.0)


def test_single_class_raises():
    with pytest.raises(ValueError):
        auc_gini_ks([1, 1, 1], [0.1, 0.2, 0.3])
