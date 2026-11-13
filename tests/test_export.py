"""The portable scorecard must reproduce the real Scorecard exactly."""
import numpy as np
import pandas as pd
import pytest

from finrisk.api.schemas import Applicant
from finrisk.explain.shap_explainer import PDExplainer
from finrisk.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from finrisk.features.woe import MISSING
from finrisk.models import export
from finrisk.models.export import PortableScorecard, applicant_frame
from finrisk.models.scorecard import Scorecard


def _data(n=9000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(50, 15, n) for c in NUMERIC_FEATURES})
    df["fico_mid"] = rng.normal(690, 30, n)
    df["annual_inc"] = np.exp(rng.normal(11, 0.5, n))
    df["dti"] = rng.gamma(4, 5, n)
    df["inq_last_6mths"] = rng.poisson(1.0, n).astype(float)
    df.loc[rng.uniform(size=n) < 0.3, "avg_cur_bal"] = np.nan            # missing values in training
    df["home_ownership"] = rng.choice(["RENT", "OWN", "MORTGAGE"], n, p=[.4, .1, .5])
    df["purpose"] = rng.choice(["debt_consolidation", "credit_card", "other"], n)
    df["verification_status"] = rng.choice(["Verified", "Not Verified"], n)
    df["addr_state"] = rng.choice(["CA", "NY", "TX", "FL"], n)
    df["initial_list_status"] = rng.choice(["w", "f"], n)
    logit = (-2.0 - 0.03 * (df.fico_mid - 690) + 0.03 * (df.dti - 18)
             + 0.25 * (df.inq_last_6mths - 1) + 0.35 * (df.home_ownership == "RENT"))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df, y


@pytest.fixture(scope="module")
def fitted():
    X, y = _data()
    sc = Scorecard(NUMERIC_FEATURES, CATEGORICAL_FEATURES, min_iv=0.005).fit(X, y)
    ps = PortableScorecard.from_scorecard(sc, X.sample(3000, random_state=0), "7", "independent",
                                          n_train=len(X), train_default_rate=float(y.mean()))
    return X, sc, ps


def test_marker_is_the_same_as_the_woe_module():
    assert export.MISSING == MISSING


def test_probabilities_scores_and_logits_match_exactly(fitted):
    X, sc, ps = fitted
    assert np.allclose(ps.predict_proba(X), sc.predict_proba(X), atol=1e-12)
    assert np.allclose(ps.score(X), sc.score(X), atol=1e-9)


def test_exact_match_survives_a_json_round_trip(fitted, tmp_path):
    X, sc, ps = fitted
    path = tmp_path / "sc.json"
    ps.save(path)
    again = PortableScorecard.load(path)
    assert np.allclose(again.predict_proba(X), sc.predict_proba(X), atol=1e-12)
    assert again.spec["model_version"] == "7"


def test_contributions_match_the_explainer_and_add_up(fitted):
    X, sc, ps = fitted
    sample = X.sample(500, random_state=1)
    bg = X.sample(3000, random_state=0)
    ref = PDExplainer(sc, bg).contributions(sample)
    got = ps.contributions(sample)
    assert np.allclose(got.to_numpy(), ref.to_numpy(), atol=1e-9)
    p = ps.predict_proba(sample)
    assert np.allclose(ps.base_value + got.sum(axis=1), np.log(p / (1 - p)), atol=1e-8)


def test_unseen_categories_and_missing_values_match_too(fitted):
    X, sc, ps = fitted
    odd = X.sample(300, random_state=2).copy()
    odd["home_ownership"] = "SOMETHING_NEW"
    odd.loc[odd.index[:100], "dti"] = np.nan
    odd.loc[odd.index[100:200], "home_ownership"] = None
    assert np.allclose(ps.predict_proba(odd), sc.predict_proba(odd), atol=1e-12)


def test_points_table_is_stored_for_the_dashboard(fitted):
    _, sc, ps = fitted
    pts = pd.DataFrame(ps.spec["points_table"])
    assert set(pts["feature"]) == set(sc.features_)
    assert {"bin", "n", "bad_rate", "woe", "points"} <= set(pts.columns)


def test_applicant_frame_matches_the_api_schema_frame():
    example = Applicant.model_config["json_schema_extra"]["example"]
    api_frame = Applicant(**example).to_frame()
    mine = applicant_frame(example, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    pd.testing.assert_frame_equal(mine[api_frame.columns], api_frame, check_dtype=False)


def test_applicant_frame_marks_unknown_values_as_missing():
    f = applicant_frame({"loan_amnt": 1000.0, "annual_inc": 0.0, "fico_score": 700.0,
                         "avg_cur_bal": None}, ["fico_mid", "avg_cur_bal", "loan_to_income"], ["purpose"])
    assert f.loc[0, "fico_mid"] == 700
    assert np.isnan(f.loc[0, "avg_cur_bal"]) and np.isnan(f.loc[0, "loan_to_income"])
    assert f.loc[0, "purpose"] is None
