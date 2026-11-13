#!/usr/bin/env python3
"""
Phase 5b for FinRisk Lab: portable scorecard export, Streamlit dashboard, Docker app files.

Run from the ROOT of your repo (after Phase 5a):

    python apply_phase5b.py            # writes the files
    python apply_phase5b.py --force    # overwrite even if you edited them

Then:
    python -m finrisk.models.export                  # writes reports/champion_scorecard.json
    streamlit run streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(".").resolve()
FORCE = "--force" in sys.argv

FILES = {}

# ---------------------------------------------------------------------------
FILES["src/finrisk/models/export.py"] = r'''"""Portable scorecard: the fitted scorecard as plain JSON, scored with numpy/pandas only.

Why: the public demo must run without a database, MLflow, scikit-learn or any model server.
The JSON holds the bins, WoE values, coefficients and scorecard points. A test proves that
it reproduces the original Scorecard exactly.

Usage:
    python -m finrisk.models.export        # champion from the registry -> reports/champion_scorecard.json
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from finrisk.validation.results import clean

MISSING = "__MISSING__"   # same marker as features/woe.py


def _apply_bins(b: dict, x: pd.Series) -> np.ndarray:
    if b["kind"] == "numeric":
        v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        idx = np.searchsorted(np.asarray(b["edges"], dtype=float), v, side="left")
        out = np.asarray(b["woe"], dtype=float)[idx]
        return np.where(np.isnan(v), b["missing_woe"], out)
    s = x.astype(object).where(x.notna(), MISSING).astype(str)
    fill = b["other_woe"] if b["other_woe"] is not None else 0.0
    return s.map(b["mapping"]).fillna(fill).to_numpy(dtype=float)


class PortableScorecard:
    def __init__(self, spec: dict):
        self.spec = spec
        self.features: list[str] = list(spec["features"])
        self.numeric: list[str] = list(spec["numeric"])
        self.categorical: list[str] = list(spec["categorical"])
        self.coef: dict = spec["coef"]
        self.intercept: float = spec["intercept"]
        self.factor: float = spec["factor"]
        self.offset: float = spec["offset"]
        self.mean_woe: dict = spec["mean_woe"]

    # ---------------------------------------------------------------- build / io
    @classmethod
    def from_scorecard(cls, sc, background: pd.DataFrame, model_version: str = "",
                       feature_set: str = "", n_train: int | None = None,
                       train_default_rate: float | None = None) -> "PortableScorecard":
        bins = {}
        for f in sc.features_:
            b = sc.woe_.bins_[f]
            bins[f] = {
                "kind": b.kind,
                "edges": b.edges.tolist() if b.kind == "numeric" else None,
                "woe": b.woe_values.tolist() if b.kind == "numeric" else None,
                "missing_woe": float(b.missing_woe),
                "mapping": b.mapping if b.kind == "categorical" else None,
                "other_woe": b.other_woe,
                "table": b.table[["bin", "n", "bad_rate", "woe"]],
            }
        mean_woe = sc.woe_.transform(background, sc.features_).mean().to_dict()
        spec = {
            "format": 1, "kind": "scorecard", "feature_set": feature_set,
            "model_version": str(model_version),
            "created": datetime.now().isoformat(timespec="seconds"),
            "features": list(sc.features_),
            "numeric": [f for f in sc.features_ if bins[f]["kind"] == "numeric"],
            "categorical": [f for f in sc.features_ if bins[f]["kind"] == "categorical"],
            "intercept": float(sc.model_.intercept_[0]),
            "coef": {f: float(c) for f, c in sc.coef_.items()},
            "factor": float(sc.factor_), "offset": float(sc.offset_),
            "bins": bins, "mean_woe": {f: float(v) for f, v in mean_woe.items()},
            "points_table": sc.scorecard_table(),
            "training": {"n_loans": n_train, "default_rate": train_default_rate},
        }
        return cls(clean(spec))

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.spec, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "PortableScorecard":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # ----------------------------------------------------------------- scoring
    def woe_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f: _apply_bins(self.spec["bins"][f], X[f]) for f in self.features},
                            index=X.index)

    def logit(self, X: pd.DataFrame) -> np.ndarray:
        W = self.woe_frame(X)
        return self.intercept + sum(self.coef[f] * W[f].to_numpy() for f in self.features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of default."""
        return 1.0 / (1.0 + np.exp(-self.logit(X)))

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Scorecard points (higher = safer)."""
        return self.offset - self.factor * self.logit(X)

    @property
    def base_value(self) -> float:
        return self.intercept + sum(self.coef[f] * self.mean_woe[f] for f in self.features)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contribution to the log-odds of default vs the average applicant."""
        W = self.woe_frame(X)
        return pd.DataFrame({f: self.coef[f] * (W[f] - self.mean_woe[f]) for f in self.features},
                            index=X.index)


def applicant_frame(inputs: dict, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """One-row frame for the given feature columns from raw form inputs (derived ones computed)."""
    d = dict(inputs)
    if d.get("fico_score") is not None:
        d["fico_mid"] = d["fico_score"]
    loan, inc = d.get("loan_amnt"), d.get("annual_inc")
    if loan and inc and inc > 0:
        d["loan_to_income"] = loan / inc
    rb = d.get("revol_bal")
    if rb is not None and not (isinstance(rb, float) and np.isnan(rb)) and inc and inc > 0:
        d["revol_bal_to_income"] = rb / inc
    row = {}
    for c in numeric:
        v = d.get(c)
        row[c] = np.nan if v is None else float(v)
    for c in categorical:
        row[c] = d.get(c)
    return pd.DataFrame([row])


def main() -> None:
    from finrisk.config import get_settings
    from finrisk.features.build import load_features
    from finrisk.models import registry
    from finrisk.models.scorecard import Scorecard

    fs = "independent"
    client = registry.setup_mlflow()
    sc = registry.load_model_object("champion", fs)
    if not isinstance(sc, Scorecard):
        raise SystemExit("The champion is not a scorecard; only scorecards can be exported.")
    version = client.get_model_version_by_alias(registry.registered_name(fs), "champion").version

    train = load_features("train")
    ps = PortableScorecard.from_scorecard(
        sc, train.sample(5000, random_state=0), version, fs,
        n_train=len(train), train_default_rate=float(train["is_default"].mean()))

    check = load_features("oot").sample(2000, random_state=1)
    diff = float(np.max(np.abs(ps.predict_proba(check) - sc.predict_proba(check))))
    if diff > 1e-9:
        raise SystemExit(f"Export does not reproduce the model (max PD difference {diff:.2e}).")

    path = get_settings().reports_dir / "champion_scorecard.json"
    ps.save(path)
    print(f"Wrote {path}  ({len(ps.features)} features: {', '.join(ps.features)})")
    print("Verified: identical to the registry scorecard on 2,000 OOT loans (max diff "
          f"{diff:.1e}).")


if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/dashboard/app.py"] = r'''"""FinRisk Lab dashboard (Streamlit).

    streamlit run streamlit_app.py

Reads only committed files in reports/ (no database, no MLflow), so it can be deployed publicly:
    champion_scorecard.json    portable scorecard   (python -m finrisk.models.export)
    validation_results.json    validation suite     (python -m finrisk.validation.run_validation)
    monitoring_results.json    monitoring simulation (python -m finrisk.monitoring.simulate)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from finrisk.explain.reason_codes import REASON_LABELS, reasons_text, top_reasons  # noqa: E402
from finrisk.models.export import PortableScorecard, applicant_frame  # noqa: E402
from finrisk.models.risk_bands import assign_risk_band, band_table  # noqa: E402

ICON = {"pass": "🟢 pass", "warn": "🟠 warn", "fail": "🔴 fail", "info": "⚪ info"}
LIGHT_ICON = {"green": "🟢", "amber": "🟠", "red": "🔴"}
PAGES = ["Score an applicant", "Validation results", "Monitoring", "About"]
DISCLAIMER = ("Demonstration model trained on historical LendingClub loans (2012-2013). "
              "Not for real credit decisions.")

PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
            "small_business", "car", "medical", "moving", "vacation", "house", "wedding",
            "renewable_energy", "other"]
STATES = ["CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "VA", "NC", "MI", "MA", "AZ",
          "WA", "CO", "MD", "OTHER"]
OPTIONAL_FIELDS = [   # (key, label, default shown to the user)
    ("revol_bal", "Revolving balance ($)", "14500"),
    ("revol_util", "Revolving utilization (%)", "71"),
    ("bc_util", "Bankcard utilization (%)", "65"),
    ("total_rev_hi_lim", "Total revolving credit limit ($)", "20000"),
    ("avg_cur_bal", "Average current balance ($)", "6000"),
    ("tot_cur_bal", "Total current balance ($)", "60000"),
    ("mort_acc", "Mortgage accounts", "0"),
    ("total_acc", "Total credit lines", "14"),
    ("open_acc", "Open credit lines", "9"),
    ("num_bc_sats", "Satisfactory bankcard accounts", "4"),
    ("delinq_2yrs", "Delinquencies, last 2 years", "0"),
    ("pub_rec", "Public derogatory records", "0"),
    ("pub_rec_bankruptcies", "Public record bankruptcies", "0"),
    ("mths_since_last_delinq", "Months since last delinquency", ""),
    ("mths_since_last_record", "Months since last public record", ""),
]


# ----------------------------------------------------------------------------- data
def reports_dir() -> Path:
    return Path(os.environ.get("FINRISK_REPORTS_DIR", _SRC.parent / "reports"))


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def _read_model(path: str, mtime: float) -> PortableScorecard:
    return PortableScorecard.load(path)


def load_report(name: str):
    p = reports_dir() / name
    return _read_json(str(p), p.stat().st_mtime) if p.exists() else None


def load_model() -> PortableScorecard | None:
    p = reports_dir() / "champion_scorecard.json"
    return _read_model(str(p), p.stat().st_mtime) if p.exists() else None


def missing_file_warning(name: str, command: str) -> None:
    st.warning(f"`reports/{name}` not found. Create it with:  `{command}`")


# ----------------------------------------------------------------------------- pages
def page_score() -> None:
    st.header("Score an applicant")
    st.caption(DISCLAIMER)
    model = load_model()
    if model is None:
        missing_file_warning("champion_scorecard.json", "python -m finrisk.models.export")
        return

    with st.form("applicant"):
        c1, c2, c3 = st.columns(3)
        loan_amnt = c1.number_input("Loan amount ($)", 1000, 40000, 15000, step=500)
        annual_inc = c2.number_input("Annual income ($)", 10000, 500000, 52000, step=1000)
        fico = c3.number_input("FICO score", 300, 850, 672)
        dti = c1.number_input("Debt-to-income (%)", 0.0, 100.0, 24.5)
        home = c2.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        inq = c3.number_input("Credit inquiries, last 6 months", 0, 30, 2)
        hist = c1.number_input("Credit history (years)", 0.0, 60.0, 9.5)
        emp = c2.number_input("Employment length (years)", 0.0, 40.0, 3.0)
        purpose = c3.selectbox("Loan purpose", PURPOSES)
        verification = c1.selectbox("Income verification", ["Verified", "Source Verified", "Not Verified"])
        state = c2.selectbox("State", STATES)
        with st.expander("More credit-file details (clear a box to mark it as unknown)"):
            cols = st.columns(3)
            raw_optional = {key: cols[i % 3].text_input(label, default)
                            for i, (key, label, default) in enumerate(OPTIONAL_FIELDS)}
        st.form_submit_button("Score applicant")

    optional, bad = {}, []
    for key, label, _ in OPTIONAL_FIELDS:
        text = raw_optional[key].strip()
        if text == "":
            optional[key] = None
            continue
        try:
            optional[key] = float(text)
        except ValueError:
            bad.append(label)
    if bad:
        st.error("Not a number: " + ", ".join(bad))
        return

    inputs = {"loan_amnt": float(loan_amnt), "annual_inc": float(annual_inc), "fico_score": float(fico),
              "dti": float(dti), "home_ownership": home, "inq_last_6mths": float(inq),
              "credit_history_years": float(hist), "emp_length_years": float(emp),
              "purpose": purpose, "verification_status": verification, "addr_state": state,
              "initial_list_status": "w", **optional}
    frame = applicant_frame(inputs, model.numeric, model.categorical)

    p = float(model.predict_proba(frame)[0])
    band = str(assign_risk_band([p])[0])
    points = float(model.score(frame)[0])
    base_pd = model.spec["training"].get("default_rate")

    m1, m2, m3 = st.columns(3)
    m1.metric("Probability of default", f"{p:.1%}",
              delta=None if base_pd is None else f"{p - base_pd:+.1%} vs training average",
              delta_color="inverse")
    m2.metric("Risk band", band)
    m3.metric("Scorecard points", f"{points:.0f}")

    contrib = model.contributions(frame).iloc[0]
    left, right = st.columns(2)
    with left:
        st.subheader("Why this score")
        reasons = top_reasons(contrib, frame.iloc[0], k=5)
        if reasons:
            for line in reasons_text(reasons):
                st.write(line)
        else:
            st.write("No material factors raise this applicant's risk.")
        st.caption("Contributions are in log-odds of default relative to the average applicant.")
    with right:
        df = pd.DataFrame({"factor": [REASON_LABELS.get(f, f) for f in contrib.index],
                           "impact": contrib.values})
        df["direction"] = np.where(df["impact"] > 0, "raises risk", "lowers risk")
        chart = (alt.Chart(df).mark_bar().encode(
            x=alt.X("impact:Q", title="log-odds of default"),
            y=alt.Y("factor:N", sort="-x", title=None),
            color=alt.Color("direction:N", scale=alt.Scale(
                domain=["raises risk", "lowers risk"], range=["#d62728", "#2ca02c"]), legend=None),
            tooltip=["factor", alt.Tooltip("impact:Q", format="+.3f")]))
        st.altair_chart(chart, width="stretch")

    st.subheader("What if the credit score were different?")
    grid = pd.concat([frame] * 25, ignore_index=True)
    grid["fico_mid"] = np.linspace(580, 820, 25)
    curve = pd.DataFrame({"FICO score": grid["fico_mid"], "PD": model.predict_proba(grid)})
    line = alt.Chart(curve).mark_line(point=True).encode(
        x="FICO score:Q", y=alt.Y("PD:Q", axis=alt.Axis(format="%")),
        tooltip=["FICO score", alt.Tooltip("PD:Q", format=".1%")])
    st.altair_chart(line, width="stretch")


def _decile_chart(deciles: pd.DataFrame):
    d = deciles.copy()
    lim = float(max(d["mean_pd"].max(), d["observed_dr"].max(), d["ci_high"].max()))
    diag = alt.Chart(pd.DataFrame({"x": [0, lim], "y": [0, lim]})).mark_line(
        strokeDash=[5, 5], color="gray").encode(x="x:Q", y="y:Q")
    bars = alt.Chart(d).mark_errorbar().encode(
        x=alt.X("mean_pd:Q", title="predicted PD (decile mean)", axis=alt.Axis(format="%")),
        y=alt.Y("ci_low:Q", title="observed default rate (95% interval)", axis=alt.Axis(format="%")),
        y2="ci_high:Q")
    pts = alt.Chart(d).mark_point(filled=True, size=70).encode(
        x="mean_pd:Q", y="observed_dr:Q",
        tooltip=[alt.Tooltip("mean_pd:Q", format=".1%"), alt.Tooltip("observed_dr:Q", format=".1%")])
    return diag + bars + pts


def page_validation() -> None:
    st.header("Validation results")
    v = load_report("validation_results.json")
    if v is None:
        missing_file_warning("validation_results.json", "python -m finrisk.validation.run_validation")
        return
    st.caption(f"Run {v['run_id']} generated {v['generated_at']}. Out-of-time test = loans issued in 2015; "
               "models were trained on 2012-2013.")
    models = list(v["models"])
    sel = st.selectbox("Model", models,
                       format_func=lambda m: f"{m}  ({v['models'][m]['role']}, version {v['models'][m]['version']})")
    res = pd.DataFrame([r for r in v["results"] if r["model"] == sel])
    head = res[~res["test"].str.endswith("_detail")]
    counts = head["status"].value_counts()
    c = st.columns(4)
    for col, key in zip(c, ["pass", "warn", "fail", "info"]):
        col.metric(ICON[key], int(counts.get(key, 0)))

    st.subheader("Findings (warn / fail)")
    flagged = head[head["status"].isin(["warn", "fail"])].copy()
    if flagged.empty:
        st.write("No warnings or failures.")
    else:
        flagged["status"] = flagged["status"].map(ICON)
        st.dataframe(flagged[["test", "metric", "value", "status", "rule"]], hide_index=True,
                     width="stretch")

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Calibration, out-of-time")
        row = next((r for r in v["results"] if r["model"] == sel and r["metric"] == "oot_calibration_ratio"), None)
        if row and row["details"].get("deciles"):
            st.altair_chart(_decile_chart(pd.DataFrame(row["details"]["deciles"])), width="stretch")
            st.caption("Points on the dashed line = well calibrated. Above it = the model under-predicts risk.")
        else:
            st.write("No decile table in this run.")
    with g2:
        st.subheader("Discrimination (Gini) by period")
        g = head[(head["test"] == "out_of_time") & head["metric"].isin(["train_gini", "valid_gini", "oot_gini"])]
        if not g.empty:
            gd = pd.DataFrame({"period": g["metric"].str.replace("_gini", "", regex=False), "Gini": g["value"]})
            st.altair_chart(alt.Chart(gd).mark_bar().encode(
                x=alt.X("period:N", sort=["train", "valid", "oot"], title=None),
                y="Gini:Q", tooltip=["period", alt.Tooltip("Gini:Q", format=".3f")]),
                width="stretch")

    recal = res[res["test"] == "recalibration"]
    if not recal.empty:
        st.subheader("Can recalibrating on 2014 repair 2015?")
        r = pd.DataFrame({"method": recal["metric"].str.replace("_oot_calibration_ratio", "", regex=False),
                          "ratio after": recal["value"],
                          "ratio before": [d.get("ratio_before") for d in recal["details"]],
                          "status": recal["status"].map(ICON)})
        st.dataframe(r, hide_index=True, width="stretch")
        st.caption("Ratio = observed / predicted default rate. 1.00 is perfect; "
                   "the validation policy tolerates 10% (warn) and 20% (fail).")

    with st.expander("All results"):
        show = res[["test", "metric", "value", "status", "rule"]].copy()
        show["status"] = show["status"].map(ICON)
        st.dataframe(show, hide_index=True, width="stretch")
    with st.expander("Validation policy (thresholds)"):
        th = pd.DataFrame([{"check": k, "warn": x["warn"], "fail": x["fail"]} for k, x in v["thresholds"].items()])
        st.dataframe(th, hide_index=True)


def _rules(values, colors):
    return [alt.Chart(pd.DataFrame({"y": [y]})).mark_rule(strokeDash=dash, color=color).encode(y="y:Q")
            for y, dash, color in zip(values, [[1, 0]] + [[4, 4]] * (len(values) - 1), colors)]


def page_monitoring() -> None:
    st.header("Portfolio monitoring simulation")
    rep = load_report("monitoring_results.json")
    if rep is None:
        missing_file_warning("monitoring_results.json", "python -m finrisk.monitoring.simulate")
        return
    p = rep["params"]
    st.caption(f"Monthly vintages 2014-2016. Reference population: {p['reference']}. Outcomes assumed visible "
               f"{p['outcome_lag_months']} months after issue; known up to {p['outcomes_known_until']} "
               "(2016 loans had not matured, so 2016 has input monitors only).")
    names = list(rep["models"])
    sel = st.selectbox("Model", names, format_func=lambda n: f"{n}  ({rep['models'][n]['role']})")
    m = rep["models"][sel]
    t = pd.DataFrame(m["monthly"])
    t["month"] = pd.to_datetime(t["month"])
    for col in ["mean_pd", "observed_default_rate", "calibration_ratio_roll3", "calibration_ratio_recal_roll3",
                "score_psi", "csi_max_used", "n_loans"]:
        t[col] = pd.to_numeric(t[col], errors="coerce")

    st.subheader("Detection timeline: when would each monitor have alerted?")
    rows = []
    for name, d in m["detection"].items():
        fa, fr = d["first_amber_or_red"], d["first_red"]
        rows.append({"monitor": name,
                     "first alert (vintage)": fa["vintage"] if fa else "never",
                     "alert visible from": fa["visible_from"] if fa else "-",
                     "first red (vintage)": fr["vintage"] if fr else "-",
                     "red months": d["months_red"], "amber months": d["months_amber"],
                     "months evaluated": d["months_evaluated"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    rec = m.get("recalibration", {})
    if rec.get("months_compared"):
        st.info(f"Lagged recalibration (intercept shift on the last known vintages) moves the "
                f"calibration ratio in {rec['last_month']} from {rec['last_ratio_raw']:.3f} to "
                f"{rec['last_ratio_recalibrated']:.3f}; it narrows the gap but does not close it.")

    st.subheader("Predicted vs observed default rate, by issue month")
    long = t.melt(id_vars="month", value_vars=["mean_pd", "observed_default_rate"],
                  var_name="series", value_name="rate").dropna()
    long["series"] = long["series"].map({"mean_pd": "predicted (mean PD)", "observed_default_rate": "observed"})
    st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
        x="month:T", y=alt.Y("rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
        color="series:N", tooltip=["month:T", "series", alt.Tooltip("rate:Q", format=".2%")]),
        width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Calibration ratio (3-month pooled)")
        cr = t.melt(id_vars="month", value_vars=["calibration_ratio_roll3", "calibration_ratio_recal_roll3"],
                    var_name="series", value_name="ratio").dropna()
        cr["series"] = cr["series"].map({"calibration_ratio_roll3": "raw model",
                                         "calibration_ratio_recal_roll3": "lagged recalibration"})
        layers = [alt.Chart(cr).mark_line(point=True).encode(
            x="month:T", y=alt.Y("ratio:Q", scale=alt.Scale(zero=False), title="observed / predicted"),
            color="series:N")]
        layers += _rules([1.0, 1.1, 1.2], ["gray", "orange", "red"])
        st.altair_chart(alt.layer(*layers), width="stretch")
        st.caption("Dashed lines: +10% (amber) and +20% (red). Only months with known outcomes.")
    with c2:
        st.subheader("Input drift: score PSI and CSI")
        dr = t.melt(id_vars="month", value_vars=["score_psi", "csi_max_used"],
                    var_name="series", value_name="value").dropna()
        dr["series"] = dr["series"].map({"score_psi": "score PSI", "csi_max_used": "worst input CSI"})
        layers = [alt.Chart(dr).mark_line(point=True).encode(x="month:T", y="value:Q", color="series:N")]
        layers += _rules([0.10, 0.25], ["orange", "red"])
        st.altair_chart(alt.layer(*layers), width="stretch")
        st.caption("Thresholds: 0.10 (amber) and 0.25 (red).")

    st.subheader("Monthly origination volume")
    st.altair_chart(alt.Chart(t).mark_bar().encode(
        x="month:T", y=alt.Y("n_loans:Q", title="loans issued"), tooltip=["month:T", "n_loans"]),
        width="stretch")
    st.caption("Volume grew several-fold while default rates rose: a candidate leading indicator "
               "that the current monitors do not use.")

    with st.expander("Monthly table"):
        show = t[["month", "n_loans", "mean_pd", "observed_default_rate", "calibration_ratio_roll3",
                  "score_psi", "csi_max_used", "csi_worst_feature"]].copy()
        show["month"] = show["month"].dt.strftime("%Y-%m")
        st.dataframe(show, hide_index=True, width="stretch")


def page_about() -> None:
    st.header("About FinRisk Lab")
    st.markdown(
        "**FinRisk Lab** trains credit-risk models on public LendingClub loans and, more importantly, "
        "**validates** them the way a bank's model-risk team would: calibration, out-of-time stability, "
        "leakage, explainability, fairness proxies and ongoing monitoring.\n\n"
        "- **Data:** LendingClub accepted loans 2007-2018Q4 (Kaggle), 36-month loans only.\n"
        "- **Split by time:** train 2012-13, validation 2014, out-of-time test 2015, monitoring 2016.\n"
        "- **Models:** a WoE logistic-regression scorecard (champion) and XGBoost / LightGBM (challengers).\n"
        "- **Scoring never uses an LLM.**\n\n"
        "**Limitations.** The data has no protected attributes, so fairness checks are proxy analyses. "
        "Loans issued in 2016 had not matured, so their outcomes are censored. This is a portfolio "
        "project, not a production credit model.")
    model = load_model()
    st.subheader("Risk bands (master scale)")
    st.dataframe(band_table(), hide_index=True)
    if model is not None:
        st.subheader("The champion scorecard")
        pts = pd.DataFrame(model.spec["points_table"])
        feat = st.selectbox("Feature", model.features)
        view = pts[pts["feature"] == feat][["bin", "n", "bad_rate", "woe", "points"]]
        st.dataframe(view.style.format({"bad_rate": "{:.1%}", "woe": "{:+.3f}", "points": "{:+.1f}"}),
                     hide_index=True, width="stretch")
        st.caption(f"Scorecard version {model.spec['model_version']}, built {model.spec['created']}. "
                   "Total score = sum of the points of the applicant's bin in each feature; higher is safer.")


def main() -> None:
    st.set_page_config(page_title="FinRisk Lab", page_icon="📊", layout="wide")
    st.sidebar.title("FinRisk Lab")
    st.sidebar.caption("Credit-risk modelling, validation and monitoring")
    page = st.sidebar.radio("Page", PAGES)
    st.sidebar.divider()
    st.sidebar.caption(DISCLAIMER)
    {"Score an applicant": page_score, "Validation results": page_validation,
     "Monitoring": page_monitoring, "About": page_about}[page]()


main()
'''

# ---------------------------------------------------------------------------
FILES["streamlit_app.py"] = r'''"""Entry point for `streamlit run streamlit_app.py` and for Streamlit Community Cloud."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "src" / "finrisk" / "dashboard" / "app.py"),
               run_name="__main__")
'''

FILES["requirements.txt"] = r'''# Dependencies of the public DASHBOARD only (used by Streamlit Community Cloud).
# The full project (training, validation, API) is installed with:  pip install -e .
streamlit>=1.50
pandas
numpy
altair
'''

# ---------------------------------------------------------------------------
FILES["docker/Dockerfile.api"] = r'''FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
# editable install keeps PROJECT_ROOT = /app, so reports/ and sql/ are found
RUN pip install --no-cache-dir -e .
COPY sql ./sql
COPY reports ./reports

EXPOSE 8000
CMD ["uvicorn", "finrisk.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
'''

FILES["docker/Dockerfile.dashboard"] = r'''FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY streamlit_app.py ./
COPY reports ./reports

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
'''

FILES[".dockerignore"] = r'''# keep Docker build contexts small (the raw data alone is 1.6 GB)
.venv
.git
data/raw
data/regulatory_docs
mlruns
mlflow.db
__pycache__
*.pyc
.pytest_cache
*.bak
notebooks
'''

FILES["docker-compose.app.yml"] = r'''# Adds the scoring API and the dashboard on top of docker-compose.yml.
#
#   docker compose -f docker-compose.yml -f docker-compose.app.yml up -d --build
#
# API:        http://localhost:8000/docs
# Dashboard:  http://localhost:8501
# (the plain `docker compose up -d` is unchanged: postgres + mlflow only)
services:
  mlflow:
    environment:
      # newer MLflow versions validate the Host header; allow the service name "mlflow"
      MLFLOW_SERVER_ALLOWED_HOSTS: "*"

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    container_name: finrisk-api
    depends_on:
      postgres:
        condition: service_healthy
      mlflow:
        condition: service_started
    environment:
      POSTGRES_HOST: postgres
      POSTGRES_PORT: "5432"
      POSTGRES_USER: ${POSTGRES_USER:-finrisk}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-finrisk}
      POSTGRES_DB: ${POSTGRES_DB:-finrisk}
      MLFLOW_TRACKING_URI: http://mlflow:5000
    ports:
      - "8000:8000"
    restart: unless-stopped

  dashboard:
    build:
      context: .
      dockerfile: docker/Dockerfile.dashboard
    container_name: finrisk-dashboard
    ports:
      - "8501:8501"
    restart: unless-stopped
'''

# ---------------------------------------------------------------------------
FILES["tests/test_export.py"] = r'''"""The portable scorecard must reproduce the real Scorecard exactly."""
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
'''

# ---------------------------------------------------------------------------
FILES["tests/test_dashboard.py"] = r'''"""Runs the Streamlit app headlessly on small generated report files."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from finrisk.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from finrisk.models.export import PortableScorecard
from finrisk.models.scorecard import Scorecard
from finrisk.monitoring.drift import detection_summary, monthly_monitoring, recalibration_summary
from finrisk.validation.calibration import calibration_table
from finrisk.validation.results import clean, make, write_json
from tests.test_export import _data
from tests.test_monitoring import _portfolio, _reference

APP = Path(__file__).resolve().parents[1] / "src" / "finrisk" / "dashboard" / "app.py"


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    d = tmp_path_factory.mktemp("reports")

    X, y = _data()
    sc = Scorecard(NUMERIC_FEATURES, CATEGORICAL_FEATURES, min_iv=0.005).fit(X, y)
    PortableScorecard.from_scorecard(sc, X.sample(2000, random_state=0), "1", "independent",
                                     n_train=len(X), train_default_rate=float(y.mean())).save(d / "champion_scorecard.json")

    p = sc.predict_proba(X)
    m = "scorecard_independent"
    rows = [
        make(m, "calibration", "oot_calibration_ratio", 1.2, "fail", 0.1, 0.2, "rule",
             {"deciles": calibration_table(y, p * 0.8)}),
        make(m, "calibration", "oot_brier", 0.12, "info"),
        make(m, "out_of_time", "train_gini", 0.30, "info"),
        make(m, "out_of_time", "valid_gini", 0.29, "info"),
        make(m, "out_of_time", "oot_gini", 0.30, "info"),
        make(m, "recalibration", "intercept_oot_calibration_ratio", 1.1, "warn", 0.1, 0.2, "rule",
             {"ratio_before": 1.2}),
        make(m, "stability", "score_psi_train_to_oot", 0.01, "pass", 0.1, 0.25, "rule"),
        make(m, "stability_detail", "csi_x_train_to_oot", 0.5, "fail", 0.1, 0.25, "rule"),
    ]
    write_json(d / "validation_results.json", "val_test", rows,
               {m: {"role": "champion", "version": "1", "kind": "scorecard", "feature_set": "independent"}})

    df, pp = _portfolio(default_multiplier=1.3)
    ref, ref_p = _reference()
    t = monthly_monitoring(df, pp, ref, ref_p, ["f1"], ["c1"], pd.Timestamp("2015-12-01"))
    report = {"generated_at": "now",
              "params": {"feature_set": "independent", "outcome_lag_months": 12, "recal_window": 6,
                         "outcomes_known_until": "2015-12", "reference": "train"},
              "models": {m: {"role": "champion", "monthly": t, "detection": detection_summary(t),
                             "recalibration": recalibration_summary(t)}}}
    import json
    (d / "monitoring_results.json").write_text(json.dumps(clean(report)), encoding="utf-8")
    return d


def _app(reports, monkeypatch, page):
    monkeypatch.setenv("FINRISK_REPORTS_DIR", str(reports))
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    if page != "Score an applicant":
        at.sidebar.radio[0].set_value(page).run()
    return at


def test_scoring_page_shows_pd_band_and_points(reports, monkeypatch):
    at = _app(reports, monkeypatch, "Score an applicant")
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert labels == ["Probability of default", "Risk band", "Scorecard points"]
    assert at.metric[0].value.endswith("%")
    assert at.metric[1].value in {"R1", "R2", "R3", "R4", "R5", "R6"}


def test_scoring_page_rejects_non_numeric_optional_fields(reports, monkeypatch):
    at = _app(reports, monkeypatch, "Score an applicant")
    box = next(t for t in at.text_input if t.label == "Revolving balance ($)")
    box.set_value("lots")
    at.button[0].click().run()          # values inside a form only apply on submit
    assert not at.exception
    assert any("Not a number" in e.value for e in at.error)


def test_validation_page_renders(reports, monkeypatch):
    at = _app(reports, monkeypatch, "Validation results")
    assert not at.exception
    assert any("Findings" in s.value for s in at.subheader)
    assert len(at.dataframe) >= 2


def test_monitoring_page_renders(reports, monkeypatch):
    at = _app(reports, monkeypatch, "Monitoring")
    assert not at.exception
    assert any("Detection timeline" in s.value for s in at.subheader)


def test_about_page_renders(reports, monkeypatch):
    at = _app(reports, monkeypatch, "About")
    assert not at.exception
    assert any("About" in h.value for h in at.header)


def test_missing_report_files_give_a_helpful_warning_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("FINRISK_REPORTS_DIR", str(tmp_path))
    for page in ["Score an applicant", "Validation results", "Monitoring"]:
        at = AppTest.from_file(str(APP), default_timeout=60).run()
        if page != "Score an applicant":
            at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
        assert any("not found" in w.value for w in at.warning), page
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
        print(f"  warn    {rel}: expected text not found, skipping this patch")


def main() -> None:
    if not (ROOT / "src" / "finrisk" / "api" / "main.py").exists():
        sys.exit("Run apply_phase5a.py first, from the repo root.")

    print(f"Applying Phase 5b in {ROOT}\n")
    for rel, content in FILES.items():
        write(rel, content)

    # API: send "/" to the interactive docs instead of a 404
    patch("src/finrisk/api/main.py",
          "from fastapi import FastAPI, Request\n",
          "from fastapi import FastAPI, Request\nfrom fastapi.responses import RedirectResponse\n")
    patch("src/finrisk/api/main.py",
          '    @app.get("/health")\n',
          '    @app.get("/", include_in_schema=False)\n    def root():\n        return RedirectResponse("/docs")\n\n'
          '    @app.get("/health")\n')

    print("\nDone. Next:")
    print("  pytest -q")
    print("  python -m finrisk.models.export")
    print("  streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()