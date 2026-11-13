#!/usr/bin/env python3
"""
Phase 4 for FinRisk Lab: portfolio monitoring simulation.

Run from the ROOT of your repo (after Phase 3b):

    python apply_phase4.py            # fills the stub files
    python apply_phase4.py --force    # overwrite even if you edited them
"""
import sys
from pathlib import Path

ROOT = Path(".").resolve()
FORCE = "--force" in sys.argv

FILES = {}

# ---------------------------------------------------------------------------
FILES["sql/04_monitoring_views.sql"] = r'''-- 04_monitoring_views.sql
-- Extends monitoring.monthly_portfolio and adds reporting views.
-- Idempotent: safe to run repeatedly (the simulation runs it before writing).

ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS share_high_risk               DOUBLE PRECISION;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS csi_max_used                  DOUBLE PRECISION;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS csi_worst_feature             TEXT;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS calibration_ratio_roll3       DOUBLE PRECISION;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS gini_roll3                    DOUBLE PRECISION;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS mean_pd_recal                 DOUBLE PRECISION;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS calibration_ratio_recal_roll3 DOUBLE PRECISION;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS status_score_psi              TEXT;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS status_csi                    TEXT;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS status_calibration            TEXT;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS status_gini                   TEXT;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS outcomes_available            BOOLEAN;

-- Observed default rate by issue month and split (outcomes of recent vintages are censored)
CREATE OR REPLACE VIEW monitoring.v_default_rate_by_vintage AS
SELECT date_trunc('month', issue_d)::date            AS month,
       split,
       COUNT(*)                                      AS n_loans,
       SUM(is_default)                               AS n_defaults,
       ROUND(AVG(is_default::numeric), 4)            AS default_rate
FROM staging.model_sample
GROUP BY 1, 2;

-- Share of loans in each risk band per issue month (score band migration)
CREATE OR REPLACE VIEW monitoring.v_band_mix_by_month AS
WITH agg AS (
    SELECT date_trunc('month', c.issue_d)::date      AS month,
           p.model_name,
           p.risk_band,
           COUNT(*)                                  AS n_loans
    FROM results.predictions p
    JOIN staging.loans_clean c USING (loan_id)
    GROUP BY 1, 2, 3
)
SELECT month, model_name, risk_band, n_loans,
       ROUND(100.0 * n_loans / SUM(n_loans) OVER (PARTITION BY month, model_name), 2) AS pct_of_month
FROM agg;

-- Most recent traffic lights per model
CREATE OR REPLACE VIEW monitoring.v_latest_status AS
SELECT DISTINCT ON (model_name)
       model_name, month, n_loans, mean_pd, score_psi, csi_max_used,
       status_score_psi, status_csi, status_calibration, status_gini
FROM monitoring.monthly_portfolio
ORDER BY model_name, month DESC;
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/monitoring/drift.py"] = r'''"""Portfolio monitoring metrics, computed per issue-month ("vintage").

For every month the bank would track:
  - volume, mean predicted PD, share of loans in the two riskiest bands
  - score PSI (PD distribution vs the training population)
  - CSI of every input the model uses; the worst one is reported
  - observed default rate, calibration ratio and Gini (pooled over 3 months) WHEN outcomes are known
  - a lagged recalibration: an intercept shift fitted only on outcomes that would already be
    observable at that time (vintage <= month - outcome_lag), applied to the new month

Outcomes are not instantly known in real life. `outcome_lag_months` models that delay, and
vintages after `outcomes_until` have no reliable outcomes at all (censored in this dataset).

Traffic lights reuse the validation thresholds: green = pass, amber = warn, red = fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finrisk.models.risk_bands import assign_risk_band
from finrisk.validation.calibration import Recalibrator
from finrisk.validation.discrimination import auc_gini_ks
from finrisk.validation.results import THRESHOLDS, grade
from finrisk.validation.stability import psi_categorical, psi_numeric

LIGHT = {"pass": "green", "warn": "amber", "fail": "red"}
ALERT_MONITORS = {   # monitor -> (status column, needs known outcomes)
    "score_psi": ("status_score_psi", False),
    "csi": ("status_csi", False),
    "calibration": ("status_calibration", True),
    "gini": ("status_gini", True),
}


def to_month(dates) -> pd.Series:
    """Calendar month start for each date."""
    return pd.to_datetime(pd.Series(np.asarray(dates))).dt.to_period("M").dt.to_timestamp()


def _light(value, warn, fail, higher_is_worse=True):
    if value is None or not np.isfinite(value):
        return None
    return LIGHT[grade(value, warn, fail, higher_is_worse)]


def _rolling_ratio(defaults: pd.Series, expected: pd.Series, window: int = 3) -> pd.Series:
    roll = lambda s: s.rolling(window, min_periods=window).sum()
    return roll(defaults) / roll(expected)


def monthly_monitoring(df: pd.DataFrame, p, ref_df: pd.DataFrame, ref_p,
                       numeric: list[str], categorical: list[str],
                       outcomes_until, lag_months: int = 12,
                       recal_window: int = 6, min_known: int = 3) -> pd.DataFrame:
    d = df.reset_index(drop=True)[["issue_d", "is_default"] + list(numeric) + list(categorical)].copy()
    d["p"] = np.asarray(p, dtype=float)
    d["month"] = to_month(d["issue_d"]).to_numpy()
    outcomes_until = pd.Timestamp(outcomes_until)
    groups = {m: g for m, g in d.groupby("month")}
    months = sorted(groups)

    wp, fp = THRESHOLDS["psi"]
    wc, fc = THRESHOLDS["calibration_ratio_dev"]
    wg, fg = THRESHOLDS["gini_floor"]

    rows = []
    for m in months:
        g = groups[m]
        has_out = m <= outcomes_until
        csi = {f: psi_numeric(ref_df[f], g[f]) for f in numeric}
        csi.update({f: psi_categorical(ref_df[f], g[f]) for f in categorical})
        worst = max(csi, key=lambda k: -1.0 if not np.isfinite(csi[k]) else csi[k])

        row = {
            "month": m, "n_loans": len(g), "mean_pd": float(g["p"].mean()),
            "share_high_risk": float(np.isin(assign_risk_band(g["p"]), ["R5", "R6"]).mean()),
            "score_psi": psi_numeric(ref_p, g["p"]),
            "csi_max_used": float(csi[worst]), "csi_worst_feature": worst,
            "outcomes_available": bool(has_out),
            "n_defaults": float(g["is_default"].sum()) if has_out else np.nan,
            "observed_default_rate": float(g["is_default"].mean()) if has_out else np.nan,
            "gini": auc_gini_ks(g["is_default"], g["p"])["gini"]
                    if has_out and g["is_default"].nunique() > 1 else np.nan,
        }

        # lagged recalibration: use only outcomes already observable at month m
        known_cut = m - pd.DateOffset(months=lag_months)
        known = [k for k in months if k <= known_cut and k <= outcomes_until]
        window = known[-recal_window:]
        row.update({"recal_window_start": pd.NaT, "recal_window_end": pd.NaT,
                    "recal_delta": np.nan, "mean_pd_recal": np.nan})
        if len(window) >= min_known:
            w = d[d["month"].isin(window)]
            rec = Recalibrator("intercept").fit(w["p"], w["is_default"])
            row.update({"recal_window_start": window[0], "recal_window_end": window[-1],
                        "recal_delta": float(rec.delta_),
                        "mean_pd_recal": float(rec.transform(g["p"]).mean())})
        rows.append(row)

    t = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)

    exp = t["mean_pd"] * t["n_loans"]
    exp_recal = t["mean_pd_recal"] * t["n_loans"]
    t["calibration_ratio_roll3"] = _rolling_ratio(t["n_defaults"], exp)
    t["calibration_ratio_recal_roll3"] = _rolling_ratio(t["n_defaults"], exp_recal)

    gini3 = []
    for i in range(len(t)):
        win = t.iloc[max(0, i - 2): i + 1]
        if len(win) < 3 or not win["outcomes_available"].all():
            gini3.append(np.nan)
            continue
        sub = d[d["month"].isin(win["month"])]
        gini3.append(auc_gini_ks(sub["is_default"], sub["p"])["gini"])
    t["gini_roll3"] = gini3

    t["status_score_psi"] = [_light(v, wp, fp) for v in t["score_psi"]]
    t["status_csi"] = [_light(v, wp, fp) for v in t["csi_max_used"]]
    t["status_calibration"] = [_light(abs(v - 1) if np.isfinite(v) else None, wc, fc)
                               for v in t["calibration_ratio_roll3"]]
    t["status_calibration_recal"] = [_light(abs(v - 1) if np.isfinite(v) else None, wc, fc)
                                     for v in t["calibration_ratio_recal_roll3"]]
    t["status_gini"] = [_light(v, wg, fg, higher_is_worse=False) for v in t["gini_roll3"]]
    return t


def detection_summary(t: pd.DataFrame, lag_months: int = 12) -> dict:
    """When would each monitor have first raised an amber / red alert, and when would it be VISIBLE?

    Input monitors (score PSI, CSI) are visible in the month itself; outcome monitors only
    `lag_months` later, once the outcomes of that vintage are observable.
    """
    out = {}
    for name, (col, needs_outcomes) in ALERT_MONITORS.items():
        s = t[["month", col]].dropna()
        lag = lag_months if needs_outcomes else 0

        def first(levels):
            hit = s[s[col].isin(levels)]
            if hit.empty:
                return None
            m = hit["month"].min()
            return {"vintage": m.strftime("%Y-%m"),
                    "visible_from": (m + pd.DateOffset(months=lag)).strftime("%Y-%m")}

        out[name] = {"first_amber_or_red": first(["amber", "red"]), "first_red": first(["red"]),
                     "months_evaluated": int(len(s)),
                     "months_red": int((s[col] == "red").sum()),
                     "months_amber": int((s[col] == "amber").sum())}
    return out


def recalibration_summary(t: pd.DataFrame) -> dict:
    """Does a lagged recalibration keep the calibration ratio closer to 1 than doing nothing?"""
    both = t.dropna(subset=["calibration_ratio_roll3", "calibration_ratio_recal_roll3"])
    if both.empty:
        return {"months_compared": 0}
    return {
        "months_compared": int(len(both)),
        "mean_abs_deviation_raw": float((both["calibration_ratio_roll3"] - 1).abs().mean()),
        "mean_abs_deviation_recalibrated": float((both["calibration_ratio_recal_roll3"] - 1).abs().mean()),
        "last_ratio_raw": float(both["calibration_ratio_roll3"].iloc[-1]),
        "last_ratio_recalibrated": float(both["calibration_ratio_recal_roll3"].iloc[-1]),
        "last_month": both["month"].iloc[-1].strftime("%Y-%m"),
    }
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/monitoring/simulate.py"] = r'''"""Monthly portfolio monitoring simulation for the champion and challenger.

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
'''

# ---------------------------------------------------------------------------
FILES["tests/test_monitoring.py"] = r'''"""Tests for the monitoring metrics (no database needed)."""
import numpy as np
import pandas as pd
import pytest

from finrisk.monitoring.drift import (
    detection_summary, monthly_monitoring, recalibration_summary, to_month)

OUTCOMES_UNTIL = pd.Timestamp("2015-12-01")
LAG = 12


def _portfolio(n_per_month=2500, start="2014-01-01", months=36, seed=0,
               default_multiplier=1.0, score_shift_from=None, score_shift=1.0):
    rng = np.random.default_rng(seed)
    month_starts = pd.date_range(start, periods=months, freq="MS")
    frames = []
    for m in month_starts:
        n = n_per_month
        p_true = rng.beta(2, 12, n)
        shifted = score_shift_from is not None and m >= pd.Timestamp(score_shift_from)
        p = np.clip(p_true * (score_shift if shifted else 1.0), 1e-4, 0.99)
        y = (rng.uniform(size=n) < np.clip(p_true * default_multiplier, 0, 1)).astype(int)
        frames.append(pd.DataFrame({
            "issue_d": m + pd.to_timedelta(rng.integers(0, 27, n), unit="D"),
            "is_default": y, "p": p,
            "f1": rng.normal(size=n), "c1": rng.choice(["a", "b", "c"], n)}))
    df = pd.concat(frames, ignore_index=True)
    return df, df["p"].to_numpy()


def _reference(seed=99):
    df, p = _portfolio(n_per_month=2500, start="2012-01-01", months=24, seed=seed)
    return df, p


def _run(**kw):
    df, p = _portfolio(**{k: v for k, v in kw.items() if k in
                          ("default_multiplier", "score_shift_from", "score_shift", "seed")})
    ref, ref_p = _reference()
    return monthly_monitoring(df, p, ref, ref_p, ["f1"], ["c1"], OUTCOMES_UNTIL,
                              lag_months=kw.get("lag", LAG))


def test_to_month_truncates_to_month_start():
    out = to_month(["2015-03-17", "2015-03-01", "2016-12-31"])
    assert list(out) == [pd.Timestamp("2015-03-01"), pd.Timestamp("2015-03-01"),
                         pd.Timestamp("2016-12-01")]


def test_stable_portfolio_is_green_and_calibrated():
    t = _run()
    assert len(t) == 36
    assert t["score_psi"].max() < 0.10
    assert (t["status_score_psi"] == "green").all()
    ratios = t["calibration_ratio_roll3"].dropna()
    assert ((ratios - 1).abs() < 0.08).all()


def test_underprediction_is_detected_by_the_calibration_monitor():
    t = _run(default_multiplier=1.4)
    ratios = t["calibration_ratio_roll3"].dropna()
    assert (ratios > 1.25).all()
    assert (t["status_calibration"].dropna() == "red").all()
    # ...while the score distribution itself did not move, so score PSI stays green
    assert (t["status_score_psi"] == "green").all()


def test_months_without_known_outcomes_have_no_outcome_metrics():
    t = _run()
    late = t[t["month"] > OUTCOMES_UNTIL]
    assert len(late) == 12
    assert not late["outcomes_available"].any()
    for col in ["observed_default_rate", "calibration_ratio_roll3", "gini_roll3", "status_calibration"]:
        assert late[col].isna().all(), col
    assert late["score_psi"].notna().all()            # input monitors still work


def test_score_drift_raises_the_psi_monitor():
    t = _run(score_shift_from="2015-07-01", score_shift=1.8)
    before = t[t["month"] < pd.Timestamp("2015-07-01")]
    after = t[t["month"] >= pd.Timestamp("2015-07-01")]
    assert (before["status_score_psi"] == "green").all()
    assert (after["status_score_psi"] == "red").all()


def test_recalibration_uses_only_outcomes_known_at_the_time():
    t = _run(default_multiplier=1.3)
    with_window = t.dropna(subset=["recal_window_end"])
    assert len(with_window) > 0
    cutoff = with_window["month"] - pd.DateOffset(months=LAG)
    assert (with_window["recal_window_end"] <= cutoff).all()
    assert (with_window["recal_window_end"] <= OUTCOMES_UNTIL).all()
    early = t[t["month"] < pd.Timestamp("2014-01-01") + pd.DateOffset(months=LAG + 2)]
    assert early["recal_window_end"].isna().all()    # not enough known history yet


def test_recalibration_helps_when_the_bias_is_stable():
    t = _run(default_multiplier=1.3)
    rec = recalibration_summary(t)
    assert rec["months_compared"] > 0
    assert rec["mean_abs_deviation_recalibrated"] < rec["mean_abs_deviation_raw"]


def test_detection_summary_applies_the_outcome_lag_to_outcome_monitors_only():
    t = _run(default_multiplier=1.4, score_shift_from="2016-01-01", score_shift=2.0)
    det = detection_summary(t, lag_months=LAG)
    cal = det["calibration"]["first_amber_or_red"]
    first_vintage = pd.Timestamp(cal["vintage"] + "-01")
    assert pd.Timestamp(cal["visible_from"] + "-01") == first_vintage + pd.DateOffset(months=LAG)
    psi = det["score_psi"]["first_amber_or_red"]
    assert psi["vintage"] == psi["visible_from"]     # input monitors are visible immediately
    assert psi["vintage"].startswith("2016-01")


def test_a_monitor_that_never_fires_reports_none():
    t = _run()
    assert detection_summary(t)["score_psi"]["first_amber_or_red"] is None
'''


# ---------------------------------------------------------------------------
def write(rel: str, content: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not FORCE:
        if len(path.read_text(encoding="utf-8").splitlines()) > 3:
            print(f"  skip    {rel}  (already edited, use --force to overwrite)")
            return
    path.write_text(content, encoding="utf-8")
    print(f"  write   {rel}")


def main() -> None:
    val = ROOT / "src" / "finrisk" / "validation" / "sensitivity.py"
    if not val.exists():
        sys.exit("Run apply_phase3b.py first, from the repo root.")

    print(f"Applying Phase 4 in {ROOT}\n")
    for rel, content in FILES.items():
        write(rel, content)

    print("\nDone. Next:")
    print("  docker compose up -d")
    print("  pytest -q")
    print("  python -m finrisk.monitoring.simulate")


if __name__ == "__main__":
    main()