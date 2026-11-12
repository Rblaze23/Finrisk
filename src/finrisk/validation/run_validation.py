"""Run the validation suite on the champion and challenger and write the results.

Outputs (single source of truth for dashboard, PDF report and LLM):
    reports/validation_results.json
    results.validation_results   (PostgreSQL)

Usage:
    python -m finrisk.validation.run_validation
    python -m finrisk.validation.run_validation --feature-sets independent with_lender
    python -m finrisk.validation.run_validation --no-db
"""
from __future__ import annotations

import argparse
import importlib
import time
from datetime import datetime

import pandas as pd

from finrisk.config import get_settings
from finrisk.db import connect
from finrisk.features.build import get_feature_columns, load_features
from finrisk.models import registry
from finrisk.models.registry import load_model_object
from finrisk.models.scorecard import Scorecard
from finrisk.validation import results as R

pd.set_option("display.width", 180)
pd.set_option("display.max_colwidth", 60)

# executed in this order; a suite without a run() function is skipped
SUITES = ["calibration", "out_of_time", "stability", "sensitivity",
          "leakage", "explanation_stability", "fairness"]


def build_contexts(feature_sets: list[str]) -> list[R.ValidationContext]:
    client = registry.setup_mlflow()
    df = load_features()
    splits = {s: df[df["split"] == s].reset_index(drop=True) for s in R.SPLITS}
    contexts = []
    for fs in feature_sets:
        num, cat = get_feature_columns(fs)
        for role in ("champion", "challenger"):
            model = load_model_object(role, fs)
            kind = "scorecard" if isinstance(model, Scorecard) else model.kind
            version = client.get_model_version_by_alias(registry.registered_name(fs), role).version
            pds = {s: model.predict_proba(splits[s]) for s in R.SPLITS}
            contexts.append(R.ValidationContext(
                model_name=f"{kind}_{fs}", kind=kind, role=role, version=str(version),
                feature_set=fs, model=model, splits=splits, pds=pds,
                numeric=num, categorical=cat))
    return contexts


def run_suites(ctx: R.ValidationContext) -> list[R.TestResult]:
    results = []
    for name in SUITES:
        mod = importlib.import_module(f"finrisk.validation.{name}")
        fn = getattr(mod, "run", None)
        if fn is None:
            print(f"    - {name}: not implemented yet, skipped")
            continue
        t0 = time.time()
        res = fn(ctx)
        results.extend(res)
        print(f"    - {name}: {len(res)} results ({time.time() - t0:.0f}s)")
    return results


def summarize(results: list[R.TestResult]) -> None:
    df = pd.DataFrame([{"model": r.model, "test": r.test, "metric": r.metric,
                        "value": r.value, "status": r.status, "detail": r.is_detail}
                       for r in results])
    head = df[~df["detail"]]
    counts = head.pivot_table(index="model", columns="status", values="metric",
                              aggfunc="count", fill_value=0)
    for col in ["pass", "warn", "fail", "info"]:
        if col not in counts:
            counts[col] = 0
    print("\n" + "=" * 100)
    print("VALIDATION SUMMARY (headline results by status; feature-level diagnostics excluded)")
    print("=" * 100)
    print(counts[["pass", "warn", "fail", "info"]].to_string())

    def table(frame):
        t = frame[frame["status"].isin(["warn", "fail"])].copy()
        t["value"] = t["value"].map(lambda v: "n/a" if v is None else f"{v:.4f}")
        t["sev"] = t["status"].map({"fail": 0, "warn": 1})
        return t.sort_values(["model", "sev", "test"])[["model", "test", "metric", "value", "status"]]

    flagged = table(head)
    print("\n" + "=" * 100)
    print("FINDINGS (warn / fail)")
    print("=" * 100)
    print(flagged.to_string(index=False) if len(flagged) else "none")

    diag = table(df[df["detail"]])
    if len(diag):
        print("\n" + "=" * 100)
        print(f"FEATURE-LEVEL DIAGNOSTICS flagged on features the model uses: {len(diag)} rows "
              "(full list in the JSON)")
        print("=" * 100)
        print(diag.head(15).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-sets", nargs="+", default=["independent"])
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args()

    connect().close()  # friendly error if Docker is down
    print("Loading models and data ...")
    contexts = build_contexts(args.feature_sets)

    results: list[R.TestResult] = []
    for ctx in contexts:
        print(f"\n[{ctx.model_name}] role={ctx.role} version={ctx.version}")
        results.extend(run_suites(ctx))

    run_id = "val_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    models = {c.model_name: {"role": c.role, "version": c.version, "kind": c.kind,
                             "feature_set": c.feature_set} for c in contexts}
    path = get_settings().reports_dir / "validation_results.json"
    R.write_json(path, run_id, results, models)
    print(f"\nWrote {len(results)} results to {path}")
    if not args.no_db:
        R.write_db(run_id, results)
        print(f"Wrote results to PostgreSQL (results.validation_results, run_id={run_id})")

    summarize(results)


if __name__ == "__main__":
    main()
