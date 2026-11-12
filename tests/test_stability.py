"""Unit tests for PSI / CSI."""
import numpy as np
import pandas as pd
import pytest

from finrisk.validation.stability import csi_table, psi_categorical, psi_label, psi_numeric


def test_identical_distributions_have_psi_near_zero():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=50_000), rng.normal(size=50_000)
    assert psi_numeric(a, b) < 0.005


def test_shifted_distribution_has_large_psi():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=50_000), rng.normal(loc=1.0, size=50_000)
    assert psi_numeric(a, b) > 0.25


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(2)
    a = rng.normal(size=50_000)
    small = psi_numeric(a, rng.normal(loc=0.2, size=50_000))
    big = psi_numeric(a, rng.normal(loc=0.8, size=50_000))
    assert 0 < small < big


def test_missing_values_are_a_separate_bin():
    rng = np.random.default_rng(3)
    a = rng.normal(size=50_000)
    b = a.copy()
    b[:20_000] = np.nan                               # 40% suddenly missing
    assert psi_numeric(a, b) > 0.25


def test_categorical_psi():
    a = np.array(["x"] * 700 + ["y"] * 300)
    assert psi_categorical(a, a) == pytest.approx(0.0, abs=1e-9)
    assert psi_categorical(a, np.array(["x"] * 300 + ["y"] * 700)) > 0.25
    assert psi_categorical(a, np.array(["z"] * 1000)) > 0.25     # unseen level


def test_csi_table_ranks_the_shifted_feature_first():
    rng = np.random.default_rng(4)
    n = 30_000
    e = pd.DataFrame({"stable": rng.normal(size=n), "shifted": rng.normal(size=n),
                      "cat": rng.choice(["a", "b"], n)})
    a = pd.DataFrame({"stable": rng.normal(size=n), "shifted": rng.normal(loc=1, size=n),
                      "cat": rng.choice(["a", "b"], n)})
    t = csi_table(e, a, ["stable", "shifted"], ["cat"])
    assert t.iloc[0]["feature"] == "shifted"
    assert t.set_index("feature").loc["stable", "label"] == "stable"


def test_psi_label_thresholds():
    assert psi_label(0.05) == "stable"
    assert psi_label(0.15) == "moderate shift"
    assert psi_label(0.40) == "significant shift"
