"""FinRisk Lab dashboard (Streamlit).

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
