"""FinRisk scoring API.

    GET  /health              liveness + loaded models + mode (mlflow vs standalone)
    GET  /model-info          models, master scale, latest validation summary
    POST /score               PD, risk band and reason codes for one applicant
    POST /agent/underwrite    Autonomous Agentic Underwriting Memo, policy check & counterfactuals
    GET  /reports/validation-pdf  Official Model Validation Report as PDF

Run locally:
    uvicorn finrisk.api.main:app --port 8000
Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from finrisk.api.schemas import Applicant, ModelScore, Reason, ScoreResponse
from finrisk.config import get_settings
from finrisk.explain.reason_codes import top_reasons
from finrisk.explain.shap_explainer import PDExplainer
from finrisk.models.export import PortableScorecard, applicant_frame
from finrisk.models.gbm import GBMModel
from finrisk.models.risk_bands import assign_risk_band, band_table
from finrisk.models.scorecard import Scorecard
from finrisk.reports.pdf_generator import generate_validation_report_pdf

FEATURE_SET = "independent"


class PortableExplainer:
    """Explainer proxy for the standalone PortableScorecard."""
    def __init__(self, model: PortableScorecard):
        self.model = model

    def contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.model.contributions(frame)


@dataclass
class LoadedModel:
    role: str
    kind: str
    version: str
    model: object
    explainer: Any


def make_loaded(role: str, model, version, background: pd.DataFrame) -> LoadedModel:
    kind = "scorecard" if isinstance(model, Scorecard) else getattr(model, "kind", "scorecard")
    return LoadedModel(role, kind, str(version), model, PDExplainer(model, background))


def load_standalone_scorecard() -> dict[str, LoadedModel]:
    """Fallback: Load portable scorecard from reports/ when MLflow is not accessible."""
    reports_dir = get_settings().reports_dir
    scorecard_path = reports_dir / "champion_scorecard.json"
    if not scorecard_path.exists():
        raise RuntimeError(f"Cannot start API: {scorecard_path} not found.")
    portable = PortableScorecard.load(scorecard_path)
    version = portable.spec.get("model_version", "1")
    return {
        "champion": LoadedModel(
            role="champion",
            kind="scorecard",
            version=str(version),
            model=portable,
            explainer=PortableExplainer(portable),
        )
    }


def load_registry_models(feature_set: str = FEATURE_SET) -> dict[str, LoadedModel]:
    """Load champion and challenger from the MLflow registry; falls back to standalone."""
    try:
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
    except Exception:
        # Fall back to portable champion scorecard
        return load_standalone_scorecard()


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
    """App factory. Pass `models` to bypass MLflow (used by tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.models = models if models is not None else load_registry_models()
        yield

    app = FastAPI(
        title="FinRisk Lab Intelligence API",
        description="Autonomous Credit Underwriting, Model Scoring, and Risk Governance Platform. "
                    "Demonstration only: not for real credit decisions.",
        version="1.1.0",
        lifespan=lifespan,
    )

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

    @app.post("/agent/underwrite")
    def agent_underwrite(applicant: Applicant, request: Request, applicant_name: str = "Borrower"):
        """Run the full Autonomous Agentic Credit Analyst underwriting loop."""
        from finrisk.agent.credit_analyst import generate_credit_memo

        loaded = request.app.state.models
        champ = loaded["champion"]
        # Ensure we have a portable model or wrapped model
        if isinstance(champ.model, PortableScorecard):
            portable_model = champ.model
        else:
            p_path = get_settings().reports_dir / "champion_scorecard.json"
            portable_model = PortableScorecard.load(p_path)

        memo = generate_credit_memo(portable_model, applicant.model_dump(), applicant_name=applicant_name)
        return {
            "applicant_name": memo.applicant_name,
            "verdict": memo.verdict,
            "risk_band": memo.risk_band,
            "probability_of_default": memo.probability_of_default,
            "scorecard_points": memo.scorecard_points,
            "reasons": memo.reasons,
            "policy_checklist": [
                {"rule": e["rule"], "detail": e["detail"], "type": e["type"]} for e in memo.policy.exceptions
            ],
            "strengths": memo.policy.strengths,
            "counterfactual": {
                "target_band": memo.counterfactual.target_band,
                "target_pd": memo.counterfactual.target_pd,
                "points_gain": memo.counterfactual.scorecard_points_gain,
                "actions": memo.counterfactual.actions,
            },
            "adverse_action_notice": memo.adverse_action_notice,
            "number_guard_status": memo.number_guard_status,
        }

    @app.get("/reports/validation-pdf")
    def download_validation_pdf():
        """Download official SR 11-7 / EBA Model Validation Report as PDF."""
        reports_dir = get_settings().reports_dir
        val_path = reports_dir / "validation_results.json"
        mon_path = reports_dir / "monitoring_results.json"
        if not val_path.exists() or not mon_path.exists():
            raise HTTPException(status_code=404, detail="Reports not found.")
        val = json.loads(val_path.read_text(encoding="utf-8"))
        mon = json.loads(mon_path.read_text(encoding="utf-8"))
        pdf_bytes = generate_validation_report_pdf(val, mon)
        return Response(content=pdf_bytes, media_type="application/pdf",
                        headers={"Content-Disposition": "attachment; filename=FinRisk_Model_Validation_Report.pdf"})

    return app


app = create_app()
