"""Tests for the leakage, sensitivity, explanation-stability and fairness helpers."""
import numpy as np
import pandas as pd
import pytest

from finrisk.validation.explanation_stability import rank_corr, top_k_overlap
from finrisk.validation.fairness import define_groups, group_calibration_table
from finrisk.validation.leakage import forbidden_overlap, single_feature_gini
from finrisk.validation.sensitivity import vintage_sensitive_features


# ---- leakage ---------------------------------------------------------------
def test_single_feature_gini_detects_a_leaky_column_and_ignores_noise():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20_000)
    df = pd.DataFrame({"leak": y + rng.normal(0, 0.05, y.size),
                       "inverse_leak": -y + rng.normal(0, 0.05, y.size),
                       "noise": rng.normal(size=y.size)})
    t = single_feature_gini(df, ["leak", "inverse_leak", "noise"], y).set_index("feature")
    assert t.loc["leak", "gini"] > 0.95
    assert t.loc["inverse_leak", "gini"] > 0.95       # direction does not matter
    assert t.loc["noise", "gini"] < 0.05


def test_single_feature_gini_handles_constant_and_all_missing_columns():
    y = np.array([0, 1] * 100)
    df = pd.DataFrame({"const": 1.0, "empty": np.nan}, index=range(len(y)))
    t = single_feature_gini(df, ["const", "empty"], y)
    assert (t["gini"] == 0).all()


def test_forbidden_overlap_finds_post_origination_columns():
    assert forbidden_overlap(["fico_mid", "total_pymnt", "recoveries"]) == ["recoveries", "total_pymnt"]
    assert forbidden_overlap(["fico_mid", "dti"]) == []


# ---- sensitivity -----------------------------------------------------------
def test_vintage_sensitive_features_flags_availability_breaks_only():
    rng = np.random.default_rng(1)
    n = 6000
    year = np.repeat([2012, 2013], n // 2)
    a = rng.normal(size=n)
    a[(year == 2012) & (rng.uniform(size=n) < 0.5)] = np.nan      # 50% missing in 2012 only
    b = rng.normal(size=n)
    b[rng.uniform(size=n) < 0.05] = np.nan                        # steady 5% missing
    df = pd.DataFrame({"issue_year": year, "a": a, "b": b})
    out = vintage_sensitive_features(df, ["a", "b"])
    assert list(out.index) == ["a"]
    assert out.loc["a", "spread"] > 0.4


# ---- explanation stability ---------------------------------------------------
def test_rank_corr_identical_and_reversed():
    a = pd.Series({"f1": 0.4, "f2": 0.3, "f3": 0.2, "f4": 0.1})
    assert rank_corr(a, a) == pytest.approx(1.0)
    assert rank_corr(a, a[::-1].set_axis(a.index)) == pytest.approx(-1.0)


def test_top_k_overlap():
    a = pd.Series({"f1": 5, "f2": 4, "f3": 3, "f4": 2, "f5": 1, "f6": 0.5})
    b = pd.Series({"f1": 5, "f2": 4, "f3": 3, "f4": 0.1, "f5": 0.2, "f6": 2})
    assert top_k_overlap(a, a, k=3) == 1.0
    assert top_k_overlap(a, b, k=4) == pytest.approx(0.75)


# ---- fairness ----------------------------------------------------------------
def test_group_calibration_flags_a_systematically_under_scored_group():
    rng = np.random.default_rng(2)
    n = 40_000
    g = pd.Series(rng.choice(["A", "B"], n))
    p_true = rng.beta(2, 12, n)
    y = (rng.uniform(size=n) < p_true).astype(int)
    p = np.where(g == "B", p_true * 0.6, p_true)      # model under-scores group B
    overall = y.mean() / p.mean()
    t = group_calibration_table(g, y, p, overall).set_index("group")
    assert abs(t.loc["B", "ratio_vs_overall"]) > 0.15
    assert abs(t.loc["A", "ratio_vs_overall"]) < abs(t.loc["B", "ratio_vs_overall"])


def test_small_groups_are_not_graded():
    g = pd.Series(["A"] * 5000 + ["tiny"] * 50)
    y = np.zeros(5050, dtype=int)
    y[::7] = 1
    p = np.full(5050, 0.14)
    t = group_calibration_table(g, y, p, 1.0).set_index("group")
    assert bool(t.loc["A", "graded"]) and not bool(t.loc["tiny", "graded"])


def test_define_groups_buckets_employment_and_income():
    df = pd.DataFrame({
        "addr_state": ["CA"] * 6 + ["NY"] * 2, "annual_inc": np.arange(1, 9) * 10_000.0,
        "home_ownership": ["RENT", "OWN", "MORTGAGE", "NONE", "RENT", "OWN", "RENT", "OWN"],
        "verification_status": ["Verified"] * 7 + [None],
        "emp_length_years": [0, 1, 4, 5, 9, 10, np.nan, 2]})
    g = define_groups(df)
    assert set(g) == {"addr_state", "income_quartile", "home_ownership",
                      "verification_status", "emp_length"}
    assert g["emp_length"].iloc[0] == "<1y" and g["emp_length"].iloc[6] == "unknown"
    assert g["home_ownership"].iloc[3] == "OTHER"
    assert g["income_quartile"].nunique() == 4
