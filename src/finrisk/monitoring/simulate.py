"""Monthly portfolio monitoring simulation for the champion and challenger.

Timeline (loan issue months, from the splits in .env):
  reference population : train (2012-2013) by default, or valid (2014) with --reference valid
  monitored vintages   : valid (2014), oot (2015), monitor (2016)
  outcomes known for   : up to December of the OOT year; 2016 outcomes are censored
                         (36-month loans issued in 2016 had not matured), so 2016 only gets
                         INPUT monitors (score PSI, CSI), not outcome monitors.

Outputs:
    monitoring.monthly_portfolio   (PostgreSQL, upserted)
    reports/monitoring_results.json

Usage:
    python -m finrisk.monitoring.simulate
    python -m finrisk.monitoring.simulate --outcome-lag-months 6
    python -m finrisk.monitoring.simulate --reference valid
    python -m finrisk.monitoring.simulate --feature-set with_lender --no-db
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from finrisk.config import get_settings
from finrisk.db import connect, execute_sql
from finrisk.features.build import get_feature_columns, load_features
from finrisk.models.registry import load_model_object, setup_mlflow
from finrisk.models.scorecard import Scorecard
from finrisk.monitoring.drift import detection_summary, monthly_monitoring, recalibration_summary
from finrisk.validation.results import clean

pd.set_option("display.width", 200)

DB_COLUMNS = [
    "month", "model_name", "n_loans", "n_defaults", "observed_default_rate", "mean_pd",
    "score_psi", "share_high_risk", "csi_max_used", "csi_worst_feature",
    "calibration_ratio_roll3", "gini_roll3", "mean_pd_recal", "calibration_ratio_recal_roll3",
    "status_score_psi", "status_csi", "status_calibration", "status_gini", "outcomes_available",
]
LETTER = {"green": "G", "amber": "A", "red": "R"}


def letter(status) -> str:
    """One-letter traffic light; missing values (None / NaN) show as '-'."""
    return LETTER.get(status, "-") if isinstance(status, str) else "-"


def write_monitoring(model_name: str, table: pd.DataFrame) -> None:
    from psycopg2.extras import execute_values

    rows = []
    for rec in table.to_dict("records"):
        rec = dict(rec, model_name=model_name, month=rec["month"].date())
        if not np.isnan(rec["n_defaults"]):
            rec["n_defaults"] = int(rec["n_defaults"])
        values = []
        for c in DB_COLUMNS:
            v = clean(rec[c]) if c != "month" else rec[c]
            values.append(v)
        rows.append(tuple(values))

    cols = ", ".join(DB_COLUMNS)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in DB_COLUMNS if c not in ("month", "model_name"))
    conn = connect()
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                f"INSERT INTO monitoring.monthly_portfolio ({cols}) VALUES %s "
                f"ON CONFLICT (month, model_name) DO UPDATE SET {updates}",
                rows)
        conn.commit()
    finally:
        conn.close()


def print_table(name: str, t: pd.DataFrame) -> None:
    v = pd.DataFrame({
        "month": t["month"].dt.strftime("%Y-%m"), "n": t["n_loans"],
        "mean_pd": t["mean_pd"].round(4), "obs_dr": t["observed_default_rate"].round(4),
        "ratio_r3": t["calibration_ratio_roll3"].round(3),
        "ratio_recal": t["calibration_ratio_recal_roll3"].round(3),
        "score_psi": t["score_psi"].round(3), "csi_max": t["csi_max_used"].round(2),
        "worst_csi_feature": t["csi_worst_feature"],
        "lights": ["".join(letter(r[c]) for c in
                           ["status_score_psi", "status_csi", "status_calibration", "status_gini"])
                   for _, r in t.iterrows()],
    })
    print(f"\n[{name}]  lights = score PSI / CSI / calibration / Gini  (G green, A amber, R red, - not available)")
    print(v.to_string(index=False))


def print_detection(name: str, det: dict, rec: dict, lag: int) -> None:
    print(f"\n[{name}] DETECTION TIMELINE (outcome monitors lag {lag} months behind the vintage)")
    for monitor, d in det.items():
        fa, fr = d["first_amber_or_red"], d["first_red"]
        if fa is None:
            print(f"  {monitor:<12} never alerted ({d['months_evaluated']} months evaluated)")
        else:
            red = f", first RED vintage {fr['vintage']} (visible {fr['visible_from']})" if fr else ""
            print(f"  {monitor:<12} first alert vintage {fa['vintage']} "
                  f"(visible {fa['visible_from']}){red}; "
                  f"{d['months_red']} red / {d['months_amber']} amber of {d['months_evaluated']} months")
    if rec.get("months_compared"):
        print(f"  recalibration (intercept shift on the last 6 KNOWN vintages): mean |ratio-1| "
              f"{rec['mean_abs_deviation_raw']:.3f} -> {rec['mean_abs_deviation_recalibrated']:.3f}; "
              f"ratio in {rec['last_month']}: {rec['last_ratio_raw']:.3f} -> {rec['last_ratio_recalibrated']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--feature-set", default="independent")
    ap.add_argument("--reference", choices=["train", "valid"], default="train",
                    help="population that PSI / CSI compare against (valid = 2014, avoids the "
                         "2012 data-availability artifact)")
    ap.add_argument("--outcome-lag-months", type=int, default=12)
    ap.add_argument("--recal-window", type=int, default=6)
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args()

    connect().close()  # friendly error if Docker is down
    s = get_settings()
    setup_mlflow()
    fs = args.feature_set
    num, cat = get_feature_columns(fs)

    print("Loading data ...")
    df = load_features()
    ref = df[df["split"] == args.reference].reset_index(drop=True)
    period = df[df["split"].isin(["valid", "oot", "monitor"])].reset_index(drop=True)
    outcomes_until = pd.Timestamp(year=s.oot_year, month=12, day=1)
    print(f"reference ({args.reference}): {len(ref):,} loans | monitored: {len(period):,} loans | "
          f"outcomes known up to {outcomes_until:%Y-%m} | outcome lag {args.outcome_lag_months} months")

    if not args.no_db:
        execute_sql((s.sql_dir / "04_monitoring_views.sql").read_text(encoding="utf-8"))

    report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "params": {"feature_set": fs, "outcome_lag_months": args.outcome_lag_months,
                         "recal_window": args.recal_window,
                         "outcomes_known_until": f"{outcomes_until:%Y-%m}",
                         "reference": args.reference},
              "models": {}}

    for role in ("champion", "challenger"):
        model = load_model_object(role, fs)
        kind = "scorecard" if isinstance(model, Scorecard) else model.kind
        name = f"{kind}_{fs}"
        used = set(getattr(model, "features_", None) or (num + cat))
        t = monthly_monitoring(
            period, model.predict_proba(period), ref, model.predict_proba(ref),
            [f for f in num if f in used], [f for f in cat if f in used],
            outcomes_until, args.outcome_lag_months, args.recal_window)
        det = detection_summary(t, args.outcome_lag_months)
        rec = recalibration_summary(t)

        print_table(name, t)
        print_detection(name, det, rec, args.outcome_lag_months)
        report["models"][name] = {"role": role, "monthly": t, "detection": det, "recalibration": rec}
        if not args.no_db:
            write_monitoring(name, t)

    path = s.reports_dir / "monitoring_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(report), indent=2), encoding="utf-8")
    print(f"\nWrote {path}")
    if not args.no_db:
        print("Wrote monitoring.monthly_portfolio (views: v_default_rate_by_vintage, "
              "v_band_mix_by_month, v_latest_status)")


if __name__ == "__main__":
    main()
