"""FinRisk scoring API.

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
from fastapi.responses import RedirectResponse

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

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

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
