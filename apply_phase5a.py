#!/usr/bin/env python3
"""
Phase 5a for FinRisk Lab: the scoring API (FastAPI).

Run from the ROOT of your repo (after Phase 4):

    python apply_phase5a.py            # fills the stub files
    python apply_phase5a.py --force    # overwrite even if you edited them

Then install the one extra test dependency:   pip install httpx
"""
import sys
from pathlib import Path

ROOT = Path(".").resolve()
FORCE = "--force" in sys.argv

FILES = {}

# ---------------------------------------------------------------------------
FILES["src/finrisk/api/schemas.py"] = r'''"""Request and response schemas for the scoring API."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from finrisk.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES

DISCLAIMER = ("Demonstration model trained on historical LendingClub loans (2012-2013). "
              "Not for real credit decisions.")

# numeric features the caller does NOT provide directly (derived or renamed)
DERIVED = {"fico_mid", "loan_to_income", "revol_bal_to_income"}


class Applicant(BaseModel):
    """One loan application. Only the first three fields are required; any field you omit is
    treated as missing, exactly as in the training data."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "loan_amnt": 15000, "annual_inc": 52000, "fico_score": 672, "dti": 24.5,
        "emp_length_years": 3, "home_ownership": "RENT", "verification_status": "Verified",
        "purpose": "debt_consolidation", "addr_state": "CA", "inq_last_6mths": 2,
        "revol_bal": 14500, "revol_util": 71.0, "total_acc": 14, "open_acc": 9,
        "credit_history_years": 9.5, "initial_list_status": "w"}})

    loan_amnt: float = Field(..., gt=0, description="Requested loan amount")
    annual_inc: float = Field(..., gt=0, description="Annual income")
    fico_score: float = Field(..., ge=300, le=850, description="FICO score at application")

    dti: float | None = Field(None, ge=0, le=1000, description="Debt-to-income ratio (%)")
    emp_length_years: float | None = Field(None, ge=0, le=60)
    delinq_2yrs: float | None = Field(None, ge=0)
    inq_last_6mths: float | None = Field(None, ge=0)
    mths_since_last_delinq: float | None = Field(None, ge=0)
    mths_since_last_record: float | None = Field(None, ge=0)
    open_acc: float | None = Field(None, ge=0)
    pub_rec: float | None = Field(None, ge=0)
    revol_bal: float | None = Field(None, ge=0)
    revol_util: float | None = Field(None, ge=0, le=1000, description="Revolving utilization (%)")
    total_acc: float | None = Field(None, ge=0)
    mort_acc: float | None = Field(None, ge=0)
    pub_rec_bankruptcies: float | None = Field(None, ge=0)
    tot_cur_bal: float | None = Field(None, ge=0)
    total_rev_hi_lim: float | None = Field(None, ge=0)
    avg_cur_bal: float | None = Field(None, ge=0)
    bc_util: float | None = Field(None, ge=0, le=1000)
    num_bc_sats: float | None = Field(None, ge=0)
    credit_history_years: float | None = Field(None, ge=0, le=100)

    home_ownership: str | None = Field(None, description="RENT | MORTGAGE | OWN | OTHER")
    verification_status: str | None = None
    purpose: str | None = None
    addr_state: str | None = Field(None, description="US state code, e.g. CA")
    initial_list_status: str | None = None

    def to_frame(self) -> pd.DataFrame:
        """One-row frame with exactly the model's feature columns (derived ones computed)."""
        d = self.model_dump()
        row = {c: np.nan for c in NUMERIC_FEATURES}
        for k, v in d.items():
            if k in NUMERIC_FEATURES and v is not None:
                row[k] = float(v)
        row["fico_mid"] = float(d["fico_score"])
        row["loan_to_income"] = d["loan_amnt"] / d["annual_inc"]
        if d.get("revol_bal") is not None:
            row["revol_bal_to_income"] = d["revol_bal"] / d["annual_inc"]
        for c in CATEGORICAL_FEATURES:
            row[c] = d.get(c)
        return pd.DataFrame([row])


class Reason(BaseModel):
    feature: str
    label: str
    value: str
    impact: float = Field(..., description="Contribution to the log-odds of default (positive = raises risk)")


class ModelScore(BaseModel):
    role: str = Field(..., description="champion | challenger")
    kind: str = Field(..., description="scorecard | xgboost | lightgbm")
    version: str
    pd: float = Field(..., description="Estimated probability of default")
    risk_band: str = Field(..., description="Master-scale band R1 (safest) .. R6")
    reasons: list[Reason] = Field(..., description="Top factors that raise this applicant's risk")


class ScoreResponse(BaseModel):
    champion: ModelScore
    challenger: ModelScore | None = None
    disclaimer: str = DISCLAIMER
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/api/main.py"] = r'''"""FinRisk scoring API.

    GET  /health        liveness + which models are loaded
    GET  /model-info    models, master scale, latest validation summary
    POST /score         PD, risk band and reason codes for one applicant

Run locally (Docker services must be up: postgres + mlflow):
    uvicorn finrisk.api.main:app --port 8000
Interactive docs: http://localhost:8000/docs

The scoring pipeline never calls an LLM.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pandas as pd
from fastapi import FastAPI, Request

from finrisk.api.schemas import Applicant, ModelScore, Reason, ScoreResponse
from finrisk.config import get_settings
from finrisk.explain.reason_codes import top_reasons
from finrisk.explain.shap_explainer import PDExplainer
from finrisk.models.gbm import GBMModel
from finrisk.models.risk_bands import assign_risk_band, band_table
from finrisk.models.scorecard import Scorecard

FEATURE_SET = "independent"


@dataclass
class LoadedModel:
    role: str
    kind: str
    version: str
    model: object
    explainer: PDExplainer


def make_loaded(role: str, model, version, background: pd.DataFrame) -> LoadedModel:
    kind = "scorecard" if isinstance(model, Scorecard) else model.kind
    return LoadedModel(role, kind, str(version), model, PDExplainer(model, background))


def load_registry_models(feature_set: str = FEATURE_SET) -> dict[str, LoadedModel]:
    """Load champion and challenger from the MLflow registry."""
    from finrisk.features.build import load_features
    from finrisk.models import registry

    client = registry.setup_mlflow()
    background = load_features("train").sample(5000, random_state=0)
    out = {}
    for role in ("champion", "challenger"):
        model = registry.load_model_object(role, feature_set)
        version = client.get_model_version_by_alias(registry.registered_name(feature_set), role).version
        out[role] = make_loaded(role, model, version, background)
    return out


def score_one(loaded: LoadedModel, frame: pd.DataFrame, k: int = 5) -> ModelScore:
    p = float(loaded.model.predict_proba(frame)[0])
    contrib = loaded.explainer.contributions(frame).iloc[0]
    reasons = top_reasons(contrib, frame.iloc[0], k=k)
    return ModelScore(role=loaded.role, kind=loaded.kind, version=loaded.version, pd=p,
                      risk_band=str(assign_risk_band([p])[0]),
                      reasons=[Reason(**r) for r in reasons])


def _validation_summary() -> dict | None:
    path = get_settings().reports_dir / "validation_results.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    counts: dict[str, dict[str, int]] = {}
    for r in data["results"]:
        if r["test"].endswith("_detail"):
            continue
        c = counts.setdefault(r["model"], {"pass": 0, "warn": 0, "fail": 0, "info": 0})
        c[r["status"]] = c.get(r["status"], 0) + 1
    return {"run_id": data["run_id"], "generated_at": data["generated_at"], "counts": counts}


def create_app(models: dict[str, LoadedModel] | None = None) -> FastAPI:
    """App factory. Pass `models` to bypass MLflow (used by the tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.models = models if models is not None else load_registry_models()
        yield

    app = FastAPI(
        title="FinRisk Lab scoring API",
        description="Probability of default, risk band and reason codes. "
                    "Demonstration only: not for real credit decisions.",
        version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "models_loaded": sorted(request.app.state.models)}

    @app.get("/model-info")
    def model_info(request: Request):
        return {
            "feature_set": FEATURE_SET,
            "models": {role: {"kind": m.kind, "version": m.version}
                       for role, m in request.app.state.models.items()},
            "risk_bands": band_table().to_dict("records"),
            "latest_validation": _validation_summary(),
        }

    @app.post("/score", response_model=ScoreResponse)
    def score(applicant: Applicant, request: Request, include_challenger: bool = False):
        frame = applicant.to_frame()
        loaded = request.app.state.models
        resp = ScoreResponse(champion=score_one(loaded["champion"], frame))
        if include_challenger and "challenger" in loaded:
            resp.challenger = score_one(loaded["challenger"], frame)
        return resp

    return app


app = create_app()
'''

# ---------------------------------------------------------------------------
FILES["tests/test_api.py"] = r'''"""API tests. They use small in-memory models, so no MLflow or database is needed."""
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
'''


# ---------------------------------------------------------------------------
def write(rel: str, content: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not FORCE:
        if len(path.read_text(encoding="utf-8").splitlines()) > 3:
            print(
                f"  skip    {rel}  (already edited, use --force to overwrite)")
            return
    path.write_text(content, encoding="utf-8")
    print(f"  write   {rel}")


def patch(rel: str, old: str, new: str) -> None:
    path = ROOT / rel
    if not path.exists():
        print(f"  missing {rel}")
        return
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"  ok      {rel}  (already patched)")
    elif old in text:
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        print(f"  patch   {rel}")
    else:
        print(
            f"  warn    {rel}: expected text not found. Add \"httpx\" to dependencies by hand.")


def main() -> None:
    if not (ROOT / "src" / "finrisk" / "monitoring" / "drift.py").exists():
        sys.exit("Run apply_phase4.py first, from the repo root.")

    print(f"Applying Phase 5a in {ROOT}\n")
    for rel, content in FILES.items():
        write(rel, content)
    patch("pyproject.toml", '"groq", "pytest",', '"groq", "pytest", "httpx",')

    print("\nDone. Next:")
    print("  pip install httpx")
    print("  pytest -q")
    print("  uvicorn finrisk.api.main:app --port 8000      (then open http://localhost:8000/docs)")


if __name__ == "__main__":
    main()
