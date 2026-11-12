"""MLflow-tracked training of every candidate model.

For each feature set (independent, with_lender) and each model
(scorecard, xgboost, lightgbm) it:
  1. fits on TRAIN (GBMs early-stop on VALID; OOT is never used for fitting)
  2. logs parameters, metrics (Gini, KS, Brier, mean PD vs observed) and artifacts
  3. registers the model in the MLflow registry
  4. writes the PDs and risk bands of every loan to results.predictions
Finally it sets the champion / challenger aliases.

Usage:
    python -m finrisk.models.train
    python -m finrisk.models.train --models scorecard xgboost --feature-sets independent
    python -m finrisk.models.train --no-predictions
"""
from __future__ import annotations

import argparse
import io
import time
import urllib.request

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from finrisk.config import get_settings
from finrisk.db import connect
from finrisk.features.build import get_feature_columns, load_features
from finrisk.models import registry
from finrisk.models.gbm import GBMModel
from finrisk.models.risk_bands import assign_risk_band
from finrisk.models.scorecard import Scorecard
from finrisk.validation.discrimination import auc_gini_ks

pd.set_option("display.width", 180)
KINDS = ["scorecard", "xgboost", "lightgbm"]
SPLITS = ["train", "valid", "oot", "monitor"]


def check_services() -> None:
    connect().close()
    uri = get_settings().mlflow_tracking_uri
    if uri.startswith("http"):
        try:
            urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=5)
        except Exception as exc:
            raise SystemExit(f"MLflow is not reachable at {uri} ({exc}).\n"
                             "Start it with: docker compose up -d")


def build_model(kind: str, feature_set: str):
    num, cat = get_feature_columns(feature_set)
    return Scorecard(num, cat) if kind == "scorecard" else GBMModel(kind, num, cat)


def fit_model(model, kind: str, splits: dict[str, pd.DataFrame]) -> None:
    tr, va = splits["train"], splits["valid"]
    if kind == "scorecard":
        model.fit(tr, tr["is_default"])
    else:
        model.fit(tr, tr["is_default"], va, va["is_default"])


def evaluate(pds: dict[str, np.ndarray], splits: dict[str, pd.DataFrame]) -> dict:
    m = {}
    for s in ["train", "valid", "oot"]:
        y, p = splits[s]["is_default"].to_numpy(), pds[s]
        for k, v in auc_gini_ks(y, p).items():
            m[f"{s}_{k}"] = v
        m[f"{s}_brier"] = float(brier_score_loss(y, p))
        m[f"{s}_mean_pd"] = float(p.mean())
        m[f"{s}_observed_dr"] = float(y.mean())
    m["gini_drop_train_to_oot"] = m["train_gini"] - m["oot_gini"]
    m["oot_calibration_ratio"] = m["oot_observed_dr"] / m["oot_mean_pd"]
    m["monitor_mean_pd"] = float(pds["monitor"].mean())
    return m


def log_params(model, kind: str, feature_set: str) -> None:
    mlflow.set_tags({"kind": kind, "feature_set": feature_set})
    s = get_settings()
    base = {"kind": kind, "feature_set": feature_set,
            "train_years": f"{s.train_start_year}-{s.train_end_year}",
            "valid_year": s.valid_year, "oot_year": s.oot_year,
            "loan_term_months": s.loan_term_months}
    if kind == "scorecard":
        base.update({"n_features": len(model.features_), "min_iv": model.min_iv,
                     "max_corr": model.max_corr, "max_bins": model.max_bins,
                     "min_bin_pct": model.min_bin_pct, "C": model.C})
        mlflow.log_text("\n".join(model.features_), "features_used.txt")
        mlflow.log_text(model.scorecard_table().to_csv(index=False), "scorecard_table.csv")
        mlflow.log_text(model.iv_.to_csv(index=False), "iv_table.csv")
    else:
        base.update({**model.params, "monotone_constraints": model.monotone,
                     "n_features": len(model.feature_names), "n_trees": model.n_trees_})
        mlflow.log_text(model.feature_importance().to_csv(index=False), "feature_importance.csv")
    mlflow.log_params({k: str(v) for k, v in base.items()})


def write_predictions(model_name: str, version: str, df: pd.DataFrame, pds: np.ndarray) -> None:
    p = np.clip(pds, 1e-6, 1 - 1e-6)
    out = pd.DataFrame({
        "loan_id": df["loan_id"].to_numpy(), "model_name": model_name,
        "model_version": str(version), "split": df["split"].to_numpy(),
        "pd": p, "risk_band": assign_risk_band(p),
    })
    buf = io.StringIO()
    out.to_csv(buf, index=False, header=False, float_format="%.8f")
    buf.seek(0)
    conn = connect()
    try:
        with conn.cursor() as cur:
            # keep only the latest version of each model in the table
            cur.execute("DELETE FROM results.predictions WHERE model_name = %s", (model_name,))
            cur.copy_expert(
                "COPY results.predictions (loan_id, model_name, model_version, split, pd, risk_band) "
                "FROM STDIN WITH (FORMAT csv)", buf)
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", nargs="+", default=KINDS, choices=KINDS)
    ap.add_argument("--feature-sets", nargs="+", default=["independent", "with_lender"])
    ap.add_argument("--no-predictions", action="store_true")
    ap.add_argument("--policy", default="benchmark", choices=["benchmark", "performance"])
    args = ap.parse_args()

    check_services()
    registry.setup_mlflow()
    mlflow.set_experiment(registry.EXPERIMENT)

    df = load_features()
    splits = {s: df[df["split"] == s].reset_index(drop=True) for s in SPLITS}
    print("rows: " + ", ".join(f"{k}={len(v):,}" for k, v in splits.items()))

    summary = []
    for fs in args.feature_sets:
        for kind in args.models:
            t0 = time.time()
            with mlflow.start_run(run_name=f"{kind}__{fs}") as run:
                model = build_model(kind, fs)
                fit_model(model, kind, splits)
                pds = {s: model.predict_proba(splits[s]) for s in SPLITS}
                metrics = evaluate(pds, splits)
                log_params(model, kind, fs)
                mlflow.log_metrics(metrics)
                version = registry.log_and_register(model, kind, fs, run.info.run_id)
                if not args.no_predictions:
                    allp = np.concatenate([pds[s] for s in SPLITS])
                    write_predictions(f"{kind}_{fs}", version,
                                      pd.concat([splits[s] for s in SPLITS], ignore_index=True), allp)
            summary.append({"feature_set": fs, "model": kind, "version": version,
                            **{k: metrics[k] for k in ["train_gini", "valid_gini", "oot_gini", "oot_ks",
                                                       "oot_brier", "oot_mean_pd", "oot_observed_dr"]}})
            print(f"  {fs:<12} {kind:<9} v{version}  oot_gini={metrics['oot_gini']:.4f}  ({time.time() - t0:.0f}s)")

    for fs in args.feature_sets:
        try:
            aliases = registry.promote(fs, args.policy)
            print(f"\n[{fs}] aliases ({args.policy}): {aliases}")
        except SystemExit as exc:
            print(f"\n[{fs}] aliases not set: {exc}")

    print("\n" + "=" * 100)
    print("MODEL COMPARISON  (fit on train 2012-13, early stopping on valid 2014, honest test on oot 2015)")
    print("=" * 100)
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print("\nMLflow UI: " + get_settings().mlflow_tracking_uri)


if __name__ == "__main__":
    main()
