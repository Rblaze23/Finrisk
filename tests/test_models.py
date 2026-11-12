"""Tests for GBM models, encoder, SHAP contributions, reason codes and risk bands."""
import numpy as np
import pandas as pd
import pytest

from finrisk.explain.reason_codes import top_reasons
from finrisk.explain.shap_explainer import PDExplainer
from finrisk.models.gbm import GBMModel, TargetOrderedEncoder
from finrisk.models.risk_bands import assign_risk_band
from finrisk.models.scorecard import Scorecard

NUM = ["fico_mid", "dti", "annual_inc"]
CAT = ["home_ownership"]
FAST = {"xgboost": dict(n_estimators=60, max_depth=3, min_child_weight=5),
        "lightgbm": dict(n_estimators=60, num_leaves=7, min_child_samples=20)}


def _data(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "fico_mid": rng.normal(690, 30, n),
        "dti": rng.gamma(4, 5, n),
        "annual_inc": np.exp(rng.normal(11, 0.5, n)),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n, p=[.4, .1, .5]),
    })
    logit = (-2.0 - 0.03 * (df.fico_mid - 690) + 0.03 * (df.dti - 18)
             - 0.5 * (np.log(df.annual_inc) - 11) + 0.4 * (df.home_ownership == "RENT"))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df, y


def test_encoder_orders_by_default_rate_and_handles_unseen():
    X = pd.DataFrame({"c": ["a"] * 200 + ["b"] * 200 + ["rare"] * 5})
    y = np.array([0] * 190 + [1] * 10 + [0] * 100 + [1] * 100 + [1] * 5)
    enc = TargetOrderedEncoder(["c"], min_count=100).fit(X, y)
    out = enc.transform(pd.DataFrame({"c": ["a", "b", "rare", "never", None]}))["c"]
    assert out.iloc[0] < out.iloc[1]            # a is safer than b
    assert out.iloc[2:].isna().all()            # rare, unseen and missing -> NaN


@pytest.mark.parametrize("kind", ["xgboost", "lightgbm"])
def test_gbm_predicts_probabilities_and_respects_monotone_constraint(kind):
    X, y = _data()
    m = GBMModel(kind, NUM, CAT, params=FAST[kind]).fit(X, y)
    p = m.predict_proba(X)
    assert ((p > 0) & (p < 1)).all()

    base = X.iloc[[0]].copy()
    grid = pd.concat([base] * 40, ignore_index=True)
    grid["fico_mid"] = np.linspace(600, 800, 40)
    pd_grid = m.predict_proba(grid)
    assert np.all(np.diff(pd_grid) <= 1e-9)     # higher FICO never raises PD


@pytest.mark.parametrize("kind", ["xgboost", "lightgbm"])
def test_gbm_shap_contributions_are_additive_in_log_odds(kind):
    X, y = _data()
    m = GBMModel(kind, NUM, CAT, params=FAST[kind]).fit(X, y)
    ex = PDExplainer(m)
    sample = X.iloc[:200]
    c = ex.contributions(sample)
    logit = np.log(m.predict_proba(sample) / (1 - m.predict_proba(sample)))
    assert np.allclose(ex.base_value + c.sum(axis=1).to_numpy(), logit, atol=1e-3)


def test_scorecard_contributions_are_exactly_additive():
    X, y = _data()
    sc = Scorecard(NUM, CAT, min_iv=0.0).fit(X, y)
    ex = PDExplainer(sc, X)
    c = ex.contributions(X.iloc[:200])
    p = sc.predict_proba(X.iloc[:200])
    assert np.allclose(ex.base_value + c.sum(axis=1).to_numpy(), np.log(p / (1 - p)), atol=1e-8)


def test_reason_codes_only_risk_increasing_sorted_and_limited():
    contrib = pd.Series({"dti": 0.5, "fico_mid": -0.7, "annual_inc": 0.2, "home_ownership": 0.1})
    values = pd.Series({"dti": 31.2, "fico_mid": 720.0, "annual_inc": 38000.0, "home_ownership": "RENT"})
    r = top_reasons(contrib, values, k=2)
    assert [x["feature"] for x in r] == ["dti", "annual_inc"]
    assert r[1]["value"] == "38,000"


def test_risk_bands_boundaries():
    b = assign_risk_band([0.0, 0.049, 0.05, 0.119, 0.12, 0.25, 0.9])
    assert list(b) == ["R1", "R1", "R2", "R3", "R4", "R6", "R6"]
