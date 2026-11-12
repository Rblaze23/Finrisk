"""MLflow model registry: champion / challenger governance.

One registered model per feature set (finrisk_pd_independent, finrisk_pd_with_lender).
Every candidate (scorecard, xgboost, lightgbm) is a VERSION of it, tagged with its kind.
Aliases 'champion' and 'challenger' point to versions.

Policies:
  benchmark   (default): champion = scorecard (regulated-style, fully explainable),
                         challenger = best gradient-boosting model by OOT Gini.
                         A challenger replaces the champion only after validation.
  performance          : champion = best OOT Gini, challenger = runner-up.

Usage:
    python -m finrisk.models.registry                       # list models
    python -m finrisk.models.registry --promote benchmark   # set aliases
"""
from __future__ import annotations

import argparse

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient

from finrisk.config import get_settings

EXPERIMENT = "finrisk-pd"
PIP_REQUIREMENTS = ["mlflow>=3,<4", "pandas", "numpy", "scikit-learn",
                    "xgboost", "lightgbm", "shap"]

pd.set_option("display.width", 160)


def registered_name(feature_set: str) -> str:
    return f"finrisk_pd_{feature_set}"


def setup_mlflow() -> MlflowClient:
    uri = get_settings().mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    return MlflowClient()


class PDModel(mlflow.pyfunc.PythonModel):
    """Uniform wrapper: DataFrame of raw features in, probability of default out."""

    def __init__(self, model, kind: str, feature_set: str):
        self.model = model
        self.kind = kind
        self.feature_set = feature_set

    def predict(self, context, model_input, params=None):
        return self.model.predict_proba(pd.DataFrame(model_input))


def log_and_register(model, kind: str, feature_set: str, run_id: str) -> str:
    """Log the model in the active run and register it. Returns the version number."""
    client = setup_mlflow()
    name = registered_name(feature_set)
    mlflow.pyfunc.log_model(
        name="model",
        python_model=PDModel(model, kind, feature_set),
        registered_model_name=name,
        pip_requirements=PIP_REQUIREMENTS,
    )
    versions = client.search_model_versions(f"name='{name}'")
    mine = [v for v in versions if getattr(v, "run_id", None) == run_id] or versions
    version = max(mine, key=lambda v: int(v.version)).version
    client.set_model_version_tag(name, version, "kind", kind)
    return str(version)


def list_models(feature_set: str) -> pd.DataFrame:
    client = setup_mlflow()
    name = registered_name(feature_set)
    try:
        aliases = client.get_registered_model(name).aliases
    except Exception:
        return pd.DataFrame()
    by_version: dict[str, list[str]] = {}
    for alias, ver in aliases.items():
        by_version.setdefault(str(ver), []).append(alias)

    rows = []
    for mv in client.search_model_versions(f"name='{name}'"):
        metrics = client.get_run(mv.run_id).data.metrics if mv.run_id else {}
        rows.append({
            "version": int(mv.version),
            "kind": (mv.tags or {}).get("kind", "?"),
            "alias": ",".join(by_version.get(str(mv.version), [])),
            "valid_gini": metrics.get("valid_gini"),
            "oot_gini": metrics.get("oot_gini"),
            "oot_ks": metrics.get("oot_ks"),
            "oot_brier": metrics.get("oot_brier"),
        })
    return pd.DataFrame(rows).sort_values("version").reset_index(drop=True)


def promote(feature_set: str = "independent", policy: str = "benchmark") -> dict:
    client = setup_mlflow()
    name = registered_name(feature_set)
    df = list_models(feature_set)
    if df.empty:
        raise SystemExit(f"No registered model {name}. Run: python -m finrisk.models.train")

    latest = df.sort_values("version").groupby("kind").tail(1)   # latest version per kind
    gbm = latest[latest["kind"] != "scorecard"].sort_values("oot_gini", ascending=False)

    if policy == "benchmark":
        sc = latest[latest["kind"] == "scorecard"]
        if sc.empty or gbm.empty:
            raise SystemExit("benchmark policy needs a scorecard and at least one GBM version")
        champion, challenger = int(sc.iloc[0]["version"]), int(gbm.iloc[0]["version"])
    elif policy == "performance":
        ranked = latest.sort_values("oot_gini", ascending=False)
        if len(ranked) < 2:
            raise SystemExit("performance policy needs at least two model kinds")
        champion, challenger = int(ranked.iloc[0]["version"]), int(ranked.iloc[1]["version"])
    else:
        raise SystemExit("policy must be 'benchmark' or 'performance'")

    client.set_registered_model_alias(name, "champion", str(champion))
    client.set_registered_model_alias(name, "challenger", str(challenger))
    return {"champion": champion, "challenger": challenger}


def load_model_object(alias: str, feature_set: str = "independent"):
    """Load the underlying fitted model (Scorecard or GBMModel) behind an alias."""
    setup_mlflow()
    try:
        loaded = mlflow.pyfunc.load_model(f"models:/{registered_name(feature_set)}@{alias}")
    except Exception as exc:
        raise SystemExit(
            f"Could not load alias '{alias}' for {feature_set}: {exc}\n"
            "Run: python -m finrisk.models.train")
    return loaded.unwrap_python_model().model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--promote", choices=["benchmark", "performance"], default=None)
    p.add_argument("--feature-set", default="independent")
    args = p.parse_args()
    if args.promote:
        res = promote(args.feature_set, args.promote)
        print(f"Aliases set ({args.promote}): {res}\n")
    df = list_models(args.feature_set)
    print(f"Registered model: {registered_name(args.feature_set)}")
    print(df.round(4).to_string(index=False) if not df.empty else "(none)")


if __name__ == "__main__":
    main()
