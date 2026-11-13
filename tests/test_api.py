"""API tests. They use small in-memory models, so no MLflow or database is needed."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from finrisk.api.main import create_app, make_loaded
from finrisk.api.schemas import Applicant
from finrisk.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from finrisk.models.gbm import GBMModel
from finrisk.models.scorecard import Scorecard

EXAMPLE = Applicant.model_config["json_schema_extra"]["example"]


def _training_data(n=9000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(50, 15, n) for c in NUMERIC_FEATURES})
    df["fico_mid"] = rng.normal(690, 30, n)
    df["annual_inc"] = np.exp(rng.normal(11, 0.5, n))
    df["dti"] = rng.gamma(4, 5, n)
    df["inq_last_6mths"] = rng.poisson(1.0, n).astype(float)
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
def client():
    X, y = _training_data()
    sc = Scorecard(NUMERIC_FEATURES, CATEGORICAL_FEATURES, min_iv=0.01).fit(X, y)
    gb = GBMModel("lightgbm", NUMERIC_FEATURES, CATEGORICAL_FEATURES,
                  params=dict(n_estimators=80, num_leaves=7, min_child_samples=30)).fit(X, y)
    bg = X.sample(2000, random_state=0)
    models = {"champion": make_loaded("champion", sc, 1, bg),
              "challenger": make_loaded("challenger", gb, 2, bg)}
    with TestClient(create_app(models)) as c:
        yield c


def test_applicant_frame_has_exactly_the_model_columns():
    frame = Applicant(**EXAMPLE).to_frame()
    assert set(frame.columns) == set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    assert frame.loc[0, "loan_to_income"] == pytest.approx(15000 / 52000)
    assert frame.loc[0, "fico_mid"] == 672
    assert np.isnan(frame.loc[0, "mort_acc"])


def test_health_lists_both_models(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "models_loaded": ["challenger", "champion"]}


def test_model_info(client):
    j = client.get("/model-info").json()
    assert j["models"]["champion"] == {"kind": "scorecard", "version": "1"}
    assert j["models"]["challenger"]["kind"] == "lightgbm"
    assert [b["band"] for b in j["risk_bands"]] == ["R1", "R2", "R3", "R4", "R5", "R6"]


def test_score_returns_pd_band_and_reasons(client):
    r = client.post("/score", json=EXAMPLE)
    assert r.status_code == 200
    champ = r.json()["champion"]
    assert 0 < champ["pd"] < 1
    assert champ["risk_band"] in {"R1", "R2", "R3", "R4", "R5", "R6"}
    assert champ["kind"] == "scorecard" and champ["role"] == "champion"
    assert len(champ["reasons"]) <= 5
    assert all(x["impact"] > 0 for x in champ["reasons"])
    assert r.json()["challenger"] is None
    assert "Not for real credit decisions" in r.json()["disclaimer"]


def test_include_challenger(client):
    r = client.post("/score?include_challenger=true", json=EXAMPLE).json()
    assert r["challenger"]["kind"] == "lightgbm"
    assert 0 < r["challenger"]["pd"] < 1


def test_only_the_three_required_fields_are_enough(client):
    r = client.post("/score", json={"loan_amnt": 8000, "annual_inc": 60000, "fico_score": 720})
    assert r.status_code == 200


@pytest.mark.parametrize("bad", [
    {"loan_amnt": -5, "annual_inc": 50000, "fico_score": 700},      # negative amount
    {"loan_amnt": 5000, "annual_inc": 0, "fico_score": 700},         # zero income
    {"loan_amnt": 5000, "annual_inc": 50000, "fico_score": 900},     # impossible FICO
    {"loan_amnt": 5000, "annual_inc": 50000},                        # missing FICO
])
def test_invalid_input_is_rejected_with_422(client, bad):
    assert client.post("/score", json=bad).status_code == 422


def test_better_credit_score_means_lower_pd_for_both_models(client):
    base = dict(EXAMPLE)
    lo = client.post("/score?include_challenger=true", json={**base, "fico_score": 620}).json()
    hi = client.post("/score?include_challenger=true", json={**base, "fico_score": 800}).json()
    assert hi["champion"]["pd"] < lo["champion"]["pd"]
    assert hi["challenger"]["pd"] < lo["challenger"]["pd"]


def test_a_risky_profile_gets_the_expected_top_reason(client):
    risky = {**EXAMPLE, "fico_score": 600, "dti": 45, "inq_last_6mths": 6}
    reasons = client.post("/score", json=risky).json()["champion"]["reasons"]
    assert reasons and reasons[0]["feature"] in {"fico_mid", "dti", "inq_last_6mths"}
