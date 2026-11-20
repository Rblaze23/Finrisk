"""FinRisk Lab - Institutional Credit Risk Intelligence & Governance Platform.

    streamlit run streamlit_app.py

Features:
  - 🧑‍💼 Agentic Credit Analyst: 1-click archetypes, multi-step policy underwriting, counterfactual optimization, interactive copilot.
  - 📊 Scorecard & Sensitivity: Portable scorecard scoring, log-odds attribution, FICO / DTI elasticity curves.
  - 🛡️ Model Validation (SR 11-7 / EBA): Calibration deciles with 95% CIs, out-of-time Gini degradation, recalibration repair.
  - 🔬 Explanation Evaluation: Monotonicity compliance, temporal attribution stability, champion-challenger alignment.
  - 📈 Monitoring & Macro Stress Testing: Vintage detection timelines, feature CSI drilldown, macro shock simulator.
  - 🤖 Guarded GenAI & Regulatory RAG: Groq LLM integration, Number Guard audit logs, regulatory retrieval (SR 11-7, EBA, IFRS 9, FCRA).
  - 📄 Executive PDF Export: 1-click download of official validation reports and borrower underwriting memos.
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

from finrisk.agent.credit_analyst import (  # noqa: E402
    ARCHETYPES,
    copilot_answer,
    evaluate_underwriting_policy,
    generate_credit_memo,
)
from finrisk.explain.evaluation import run_full_explanation_audit  # noqa: E402
from finrisk.explain.reason_codes import REASON_LABELS, reasons_text, top_reasons  # noqa: E402
from finrisk.genai.rag import answer_regulatory_question, query_regulations  # noqa: E402
from finrisk.models.export import PortableScorecard, applicant_frame  # noqa: E402
from finrisk.models.risk_bands import assign_risk_band, band_table  # noqa: E402
from finrisk.reports.pdf_generator import (  # noqa: E402
    generate_underwriting_memo_pdf,
    generate_validation_report_pdf,
)

ICON = {"pass": "🟢 PASS", "warn": "🟠 WARN", "fail": "🔴 FAIL", "info": "⚪ INFO"}
PAGES = [
    "🧑‍💼 Agentic Credit Analyst",
    "📊 Applicant Scoring & Sensitivity",
    "🛡️ Model Validation (SR 11-7)",
    "🔬 Explanation Evaluation",
    "📈 Monitoring & Stress Testing",
    "🤖 Guarded GenAI & Regulatory RAG",
    "📄 Executive PDF Reports",
    "ℹ️ Architecture & Methodology",
]
DISCLAIMER = (
    "Demonstration model trained on historical LendingClub loans (2012-2013). "
    "For institutional demonstration & model risk governance research only."
)

PURPOSES = [
    "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
    "small_business", "car", "medical", "moving", "vacation", "house", "wedding",
    "renewable_energy", "other",
]
STATES = [
    "CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "VA", "NC", "MI", "MA", "AZ",
    "WA", "CO", "MD", "OTHER",
]
OPTIONAL_FIELDS = [
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

# -----------------------------------------------------------------------------
# Data Loaders
# -----------------------------------------------------------------------------
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
    st.warning(f"`reports/{name}` not found. Create it with: `{command}`")


def get_active_llm():
    """Return GroqLLM if configured and available, else None."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        return None
    try:
        from finrisk.genai.groq_client import GroqLLM
        model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        return GroqLLM(api_key=api_key, model=model_name)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
        .metric-card {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .badge-approved { background-color: #dcfce7; color: #15803d; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
        .badge-warn { background-color: #fef3c7; color: #b45309; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
        .badge-declined { background-color: #fee2e2; color: #b91c1c; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
        .stButton>button { border-radius: 8px; font-weight: 500; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Page 1: Agentic Credit Analyst
# -----------------------------------------------------------------------------
def page_agentic_analyst() -> None:
    st.title("🧑‍💼 Autonomous Agentic Credit Analyst")
    st.caption("Multi-step credit underwriting loop: Scoring -> Policy Gates -> Counterfactuals -> Underwriting Memo.")

    model = load_model()
    if model is None:
        missing_file_warning("champion_scorecard.json", "python -m finrisk.models.export")
        return

    st.subheader("1. Quick-Load Applicant Archetypes")
    cols = st.columns(4)
    if "current_inputs" not in st.session_state:
        st.session_state["current_inputs"] = ARCHETYPES["jordan_taylor_near_prime"]["inputs"].copy()
        st.session_state["current_name"] = "Jordan Taylor"

    for i, (key, arch) in enumerate(ARCHETYPES.items()):
        if cols[i].button(f"{arch['badge']}\n{arch['name']}", key=f"btn_{key}", use_container_width=True):
            st.session_state["current_inputs"] = arch["inputs"].copy()
            st.session_state["current_name"] = arch["name"]
            st.rerun()

    cur_inputs = st.session_state["current_inputs"]
    cur_name = st.session_state.get("current_name", "Borrower")

    with st.expander("📝 Inspect / Edit Applicant Financial Profile", expanded=False):
        c1, c2, c3 = st.columns(3)
        applicant_name = c1.text_input("Applicant Name", cur_name)
        loan_amnt = c2.number_input("Requested Loan ($)", 1000, 40000, int(cur_inputs.get("loan_amnt", 15000)), step=500)
        annual_inc = c3.number_input("Annual Income ($)", 10000, 500000, int(cur_inputs.get("annual_inc", 52000)), step=1000)
        fico = c1.number_input("FICO Credit Score", 300, 850, int(cur_inputs.get("fico_score", 672)))
        dti = c2.number_input("Debt-to-Income (%)", 0.0, 100.0, float(cur_inputs.get("dti", 24.5)))
        home = c3.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"],
                            index=["RENT", "MORTGAGE", "OWN", "OTHER"].index(cur_inputs.get("home_ownership", "RENT")))
        inq = c1.number_input("Inquiries (6 months)", 0, 30, int(cur_inputs.get("inq_last_6mths", 2)))
        hist = c2.number_input("Credit History (years)", 0.0, 60.0, float(cur_inputs.get("credit_history_years", 9.5)))
        purpose = c3.selectbox("Loan Purpose", PURPOSES,
                               index=PURPOSES.index(cur_inputs.get("purpose", "debt_consolidation")) if cur_inputs.get("purpose") in PURPOSES else 0)

        # Update session inputs
        cur_inputs.update({
            "loan_amnt": float(loan_amnt), "annual_inc": float(annual_inc), "fico_score": float(fico),
            "dti": float(dti), "home_ownership": home, "inq_last_6mths": float(inq),
            "credit_history_years": float(hist), "purpose": purpose,
        })

    active_llm = get_active_llm()
    # Run Autonomous Underwriting
    memo = generate_credit_memo(model, cur_inputs, applicant_name=cur_name, llm=active_llm)

    st.markdown("---")
    st.subheader(f"2. Agent Underwriting Verdict: {cur_name}")

    m1, m2, m3, m4 = st.columns(4)
    verdict_colors = {
        "APPROVED": "normal",
        "CONDITIONALLY APPROVED": "off",
        "REFERRED TO CREDIT COMMITTEE": "inverse",
        "DECLINED": "inverse",
    }
    m1.metric("Underwriting Verdict", memo.verdict)
    m2.metric("Assigned Master Scale Band", memo.risk_band)
    m3.metric("Probability of Default (PD)", f"{memo.probability_of_default:.1%}")
    m4.metric("Scorecard Points", f"{memo.scorecard_points:.0f}")

    # Tabs for detailed agent sections
    t_summary, t_counterfactual, t_policy, t_copilot = st.tabs([
        "📋 Underwriting Memo",
        "🎯 Counterfactual Path-to-Approval",
        "⚖️ Policy Checklist & Adverse Action",
        "💬 Interactive Credit Copilot",
    ])

    with t_summary:
        st.markdown(memo.memo_markdown)
        st.caption(f"🛡️ Number Guard Status: `{memo.number_guard_status}`")

    with t_counterfactual:
        st.markdown("### Actionable Counterfactual Optimization")
        c = memo.counterfactual
        st.write(
            f"**Target Objective:** Upgrade from **{c.current_band}** to **{c.target_band}** "
            f"(Target PD: `{c.target_pd:.1%}`, Scorecard Points Gain: `+{c.scorecard_points_gain:.0f}`)."
        )
        for act in c.actions:
            st.info(f"**Recommended Action:** {act['action']}\n\n*Impact:* {act.get('impact', '')}")

    with t_policy:
        st.markdown("### Bank Underwriting Policy Checklist")
        if memo.policy.exceptions:
            for e in memo.policy.exceptions:
                st.warning(f"⚠️ **{e['rule']}**: {e['detail']}")
        else:
            st.success("🟢 All standard credit policy and qualified mortgage limits satisfied.")

        if memo.adverse_action_notice:
            st.markdown("### Official FCRA / ECOA Adverse Action Statement")
            st.code(memo.adverse_action_notice, language="text")

    with t_copilot:
        st.markdown("### Interactive Credit Copilot & Conversational Underwriting")
        st.caption("Interrogate the AI credit officer regarding risk drivers, what-if scenarios, policy mitigants, and counterfactuals.")

        if active_llm is not None:
            st.success(f"🟢 **Live LLM Active:** Groq (`{active_llm.model}`) with Number Guard verification.")
        else:
            st.info("🧠 **Intelligent Credit Reasoning Engine (Offline Mode)**: Delivering contextual financial analysis without external API dependencies. Enter a free Groq API key in the sidebar for live LLM mode.")

        # Initialize session chat history if not present or switched applicant
        if "copilot_chat" not in st.session_state or st.session_state.get("copilot_applicant") != cur_name:
            st.session_state["copilot_applicant"] = cur_name
            top_f = memo.reasons[0]['label'] if memo.reasons else 'debt leverage'
            st.session_state["copilot_chat"] = [
                {
                    "role": "assistant",
                    "content": (
                        f"Hello, I am your Credit Risk Underwriting Copilot. I have analyzed **{cur_name}**'s credit file. "
                        f"Final Determination: **{memo.verdict}** (Tier **{memo.risk_band}**, Probability of Default: **{memo.probability_of_default:.1%}**, "
                        f"Scorecard Points: **{memo.scorecard_points:.0f}**). "
                        f"The primary risk factor is `{top_f}`. How can I assist you with this credit decision?"
                    )
                }
            ]

        # Display conversational chat history
        for msg in st.session_state["copilot_chat"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Quick question chips
        st.markdown("**Quick Inquiries:**")
        chip_cols = st.columns(4)
        quick_q = None
        if chip_cols[0].button("🔍 Why this verdict?", key="chip_why", use_container_width=True):
            quick_q = "Why was this applicant flagged and assigned this risk band?"
        if chip_cols[1].button("📈 What if income +15%?", key="chip_inc", use_container_width=True):
            quick_q = "What happens to the default probability if stated income increases by 15%?"
        if chip_cols[2].button("🎯 How to upgrade tier?", key="chip_upgrade", use_container_width=True):
            quick_q = "What specific counterfactual actions can this borrower take to qualify for an upgrade?"
        if chip_cols[3].button("🤝 Could a co-signer help?", key="chip_cosign", use_container_width=True):
            quick_q = "Could adding a creditworthy co-signer or collateral overturn a decline?"

        # Chat input bar
        user_input = st.chat_input("Ask Credit Copilot a question about this applicant...")
        prompt_to_run = quick_q or user_input

        if prompt_to_run:
            st.session_state["copilot_chat"].append({"role": "user", "content": prompt_to_run})
            with st.chat_message("user"):
                st.markdown(prompt_to_run)

            with st.spinner("Analyzing credit risk dynamics..."):
                response = copilot_answer(prompt_to_run, memo, cur_inputs, llm=active_llm)

            st.session_state["copilot_chat"].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
            st.rerun()


# -----------------------------------------------------------------------------
# Page 2: Applicant Scoring & Sensitivity
# -----------------------------------------------------------------------------
def page_score() -> None:
    st.title("📊 Applicant Scoring & Factor Sensitivity")
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
        with st.expander("More credit-file details"):
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
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Marginal Elasticity: FICO Sensitivity Curve")
    grid = pd.concat([frame] * 25, ignore_index=True)
    grid["fico_mid"] = np.linspace(580, 820, 25)
    curve = pd.DataFrame({"FICO score": grid["fico_mid"], "PD": model.predict_proba(grid)})
    line = alt.Chart(curve).mark_line(point=True).encode(
        x="FICO score:Q", y=alt.Y("PD:Q", axis=alt.Axis(format="%")),
        tooltip=["FICO score", alt.Tooltip("PD:Q", format=".1%")])
    st.altair_chart(line, use_container_width=True)


# -----------------------------------------------------------------------------
# Page 3: Model Validation (SR 11-7)
# -----------------------------------------------------------------------------
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
    st.title("🛡️ Model Validation Suite (SR 11-7 / EBA)")
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

    st.subheader("Validation Findings (warn / fail)")
    flagged = head[head["status"].isin(["warn", "fail"])].copy()
    if flagged.empty:
        st.write("No warnings or failures.")
    else:
        flagged["status"] = flagged["status"].map(ICON)
        st.dataframe(flagged[["test", "metric", "value", "status", "rule"]], hide_index=True, use_container_width=True)

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Calibration, Out-of-Time Deciles")
        row = next((r for r in v["results"] if r["model"] == sel and r["metric"] == "oot_calibration_ratio"), None)
        if row and row["details"].get("deciles"):
            st.altair_chart(_decile_chart(pd.DataFrame(row["details"]["deciles"])), use_container_width=True)
            st.caption("Points on the dashed line = well calibrated. Above it = model under-predicts default rate.")
        else:
            st.write("No decile table in this run.")
    with g2:
        st.subheader("Discrimination (Gini) Degradation")
        g = head[(head["test"] == "out_of_time") & head["metric"].isin(["train_gini", "valid_gini", "oot_gini"])]
        if not g.empty:
            gd = pd.DataFrame({"period": g["metric"].str.replace("_gini", "", regex=False), "Gini": g["value"]})
            st.altair_chart(alt.Chart(gd).mark_bar().encode(
                x=alt.X("period:N", sort=["train", "valid", "oot"], title=None),
                y="Gini:Q", tooltip=["period", alt.Tooltip("Gini:Q", format=".3f")]),
                use_container_width=True)

    recal = res[res["test"] == "recalibration"]
    if not recal.empty:
        st.subheader("Recalibration Repair Analysis (Can 2014 repair 2015?)")
        r = pd.DataFrame({"method": recal["metric"].str.replace("_oot_calibration_ratio", "", regex=False),
                          "ratio after": recal["value"],
                          "ratio before": [d.get("ratio_before") for d in recal["details"]],
                          "status": recal["status"].map(ICON)})
        st.dataframe(r, hide_index=True, use_container_width=True)
        st.caption("Ratio = observed / predicted default rate. 1.00 is perfect.")


# -----------------------------------------------------------------------------
# Page 4: Explanation Evaluation
# -----------------------------------------------------------------------------
def page_explanation_evaluation() -> None:
    st.title("🔬 Explanation Evaluation & Governance Suite")
    st.caption("Auditing explanation monotonicity, temporal rank stability, and champion-challenger alignment.")

    model = load_model()
    val = load_report("validation_results.json")
    if model is None:
        missing_file_warning("champion_scorecard.json", "python -m finrisk.models.export")
        return

    audit = run_full_explanation_audit(model.spec, val)

    c1, c2, c3 = st.columns(3)
    c1.metric("Explanation Audit Score", f"{audit.overall_score:.1f} / 100")
    c2.metric("Temporal Rank Correlation", f"{audit.temporal_stability['train_to_oot_rank_corr']:.3f}")
    c3.metric("Perturbation Robustness", f"{audit.robustness_index:.1%}")

    st.info(f"**Audit Finding:** {audit.summary}")

    st.subheader("1. Economic Monotonicity Compliance")
    st.caption("Confirming that feature contributions adhere to financial risk monotonicity.")
    mono_rows = []
    for m in audit.monotonicity_results:
        mono_rows.append({
            "Feature": m.feature,
            "Expected Constraint": m.expected_direction,
            "Monotonicity Status": "🟢 Compliant" if m.is_monotonic else "🔴 Violation",
            "Compliance Score": f"{m.compliance_score:.1%}",
            "Economic Logic": m.description,
        })
    st.dataframe(pd.DataFrame(mono_rows), hide_index=True, use_container_width=True)

    st.subheader("2. Champion vs Challenger Alignment")
    st.caption("Where does the tree ensemble (XGBoost) diverge from the linear scorecard?")
    align = audit.model_alignment
    st.write(align.get("divergence_insight", ""))
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Champion Top Drivers:**")
        for f in align.get("champion_top5", []):
            st.write(f"- `{f}` ({REASON_LABELS.get(f, f)})")
    with col2:
        st.markdown("**Challenger Top Drivers:**")
        for f in align.get("challenger_top5", []):
            st.write(f"- `{f}` ({REASON_LABELS.get(f, f)})")


# -----------------------------------------------------------------------------
# Page 5: Portfolio Monitoring & Stress Testing
# -----------------------------------------------------------------------------
def _rules(values, colors):
    return [alt.Chart(pd.DataFrame({"y": [y]})).mark_rule(strokeDash=dash, color=color).encode(y="y:Q")
            for y, dash, color in zip(values, [[1, 0]] + [[4, 4]] * (len(values) - 1), colors)]


def page_monitoring() -> None:
    st.title("📈 Portfolio Monitoring & Macro Stress Testing")
    rep = load_report("monitoring_results.json")
    if rep is None:
        missing_file_warning("monitoring_results.json", "python -m finrisk.monitoring.simulate")
        return
    p = rep["params"]
    st.caption(f"Monthly vintages 2014-2016. Reference population: {p['reference']}. Outcomes assumed visible "
               f"{p['outcome_lag_months']} months after issue.")

    names = list(rep["models"])
    sel = st.selectbox("Model", names, format_func=lambda n: f"{n}  ({rep['models'][n]['role']})")
    m = rep["models"][sel]
    t = pd.DataFrame(m["monthly"])
    t["month"] = pd.to_datetime(t["month"])
    for col in ["mean_pd", "observed_default_rate", "calibration_ratio_roll3", "calibration_ratio_recal_roll3",
                "score_psi", "csi_max_used", "n_loans"]:
        t[col] = pd.to_numeric(t[col], errors="coerce")

    # Portfolio KPI Ribbon
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vintages Monitored", f"{len(t)} months")
    c2.metric("Cumulative Originations", f"{t['n_loans'].sum():,.0f}")
    c3.metric("Baseline Mean PD", f"{t['mean_pd'].mean():.1%}")
    c4.metric("Worst Observed CSI", f"{t['csi_max_used'].max():.3f}", delta="Alert > 0.25", delta_color="inverse")

    st.subheader("1. Detection Timeline: The 12-Month Outcome Lag Problem")
    rows = []
    for name, d in m["detection"].items():
        fa, fr = d["first_amber_or_red"], d["first_red"]
        rows.append({"monitor": name,
                     "first alert (vintage)": fa["vintage"] if fa else "never",
                     "alert visible from": fa["visible_from"] if fa else "-",
                     "first red (vintage)": fr["vintage"] if fr else "-",
                     "red months": d["months_red"], "amber months": d["months_amber"],
                     "months evaluated": d["months_evaluated"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("2. Predicted vs Observed Default Rate Drift")
    long = t.melt(id_vars="month", value_vars=["mean_pd", "observed_default_rate"],
                  var_name="series", value_name="rate").dropna()
    long["series"] = long["series"].map({"mean_pd": "predicted (mean PD)", "observed_default_rate": "observed"})
    st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
        x="month:T", y=alt.Y("rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
        color="series:N", tooltip=["month:T", "series", alt.Tooltip("rate:Q", format=".2%")]),
        use_container_width=True)

    st.subheader("3. Interactive Macroeconomic Stress-Testing Simulator")
    st.caption("Simulate macroeconomic shocks on portfolio default probability and Expected Credit Loss (ECL).")
    sc1, sc2, sc3 = st.columns(3)
    unemp_shock = sc1.slider("Unemployment Shock (+%)", 0.0, 5.0, 1.5, step=0.5)
    rate_shock = sc2.slider("Fed Funds Rate Shock (+bps)", 0, 400, 150, step=25)
    income_shock = sc3.slider("Real Income Contraction (-%)", 0.0, 15.0, 5.0, step=1.0)

    # Simplified macro-credit sensitivity elasticity
    base_pd = float(t["mean_pd"].iloc[-1])
    shocked_pd = min(0.40, base_pd * (1.0 + (unemp_shock * 0.08) + (rate_shock / 1000.0 * 0.05) + (income_shock * 0.03)))
    ead = float(t["n_loans"].iloc[-1] * 15000.0)
    lgd = 0.45
    baseline_loss = base_pd * lgd * ead
    stressed_loss = shocked_pd * lgd * ead

    k1, k2, k3 = st.columns(3)
    k1.metric("Stressed Mean PD", f"{shocked_pd:.1%}", delta=f"{shocked_pd - base_pd:+.2%}", delta_color="inverse")
    k2.metric("Active Portfolio EAD", f"${ead / 1e6:.1f}M")
    k3.metric("Projected Stressed Loss (ECL)", f"${stressed_loss / 1e6:.2f}M", delta=f"+${(stressed_loss - baseline_loss)/1e6:.2f}M", delta_color="inverse")


# -----------------------------------------------------------------------------
# Page 6: Guarded GenAI & Regulatory RAG
# -----------------------------------------------------------------------------
def page_genai_rag() -> None:
    st.title("🤖 Guarded GenAI & Regulatory Intelligence Hub")
    st.caption("Verifiable LLM report synthesis, Number Guard audit logs, and Regulatory RAG (SR 11-7, EBA, IFRS 9).")

    t_rag, t_guard = st.tabs(["📚 Regulatory RAG Engine", "🛡️ Number Guard Audit Logs"])

    with t_rag:
        st.subheader("Supervisory Standards Knowledge Retrieval")
        active_llm = get_active_llm()
        if active_llm is not None:
            st.caption(f"🟢 Grounded RAG synthesized by Live Groq LLM (`{active_llm.model}`).")
        else:
            st.caption("🧠 Grounded RAG running via BM25 Semantic Retrieval Engine.")

        reg_query = st.text_input("Enter compliance or model risk question:",
                                  "What are the ongoing monitoring requirements under Fed SR 11-7?")
        if st.button("Search Regulations", key="btn_reg_search"):
            ans = answer_regulatory_question(reg_query, llm=active_llm)
            st.markdown(f"### Regulatory Finding\n{ans['answer']}")
            st.markdown("### Grounded Citations")
            for src in ans["sources"]:
                st.info(f"**{src['standard']}** — {src['section']}\n*{src['citation']}*\n\n{src['text']}")

    with t_guard:
        st.subheader("Number Guard Verification Architecture")
        st.markdown(
            "Every quantitative claim written by the LLM is deterministically extracted and cross-checked "
            "against `validation_results.json` and `monitoring_results.json`. Unsupported numbers are rejected."
        )
        audit_rep = load_report("validation_report_audit.json")
        if audit_rep:
            st.json(audit_rep)
        else:
            st.info("Validation report audit log is ready to be inspected upon report generation.")


# -----------------------------------------------------------------------------
# Page 7: Executive PDF Reports
# -----------------------------------------------------------------------------
def page_pdf_reports() -> None:
    st.title("📄 Bank-Grade PDF Report Generator")
    st.caption("Download official, print-ready validation reports and borrower underwriting memos.")

    val = load_report("validation_results.json")
    mon = load_report("monitoring_results.json")
    model = load_model()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. Model Validation Report (SR 11-7)")
        st.write("Complete institutional Model Risk Management audit report with calibration tables and verdicts.")
        if val and mon:
            pdf_bytes = generate_validation_report_pdf(val, mon)
            st.download_button(
                label="📄 Download Model Validation Report (PDF)",
                data=pdf_bytes,
                file_name="FinRisk_Model_Validation_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.warning("Validation reports not found.")

    with c2:
        st.subheader("2. Borrower Underwriting Memo")
        st.write("Credit Committee underwriting memo with vitals, counterfactuals, and Adverse Action notice.")
        if model:
            sample_inputs = ARCHETYPES["jordan_taylor_near_prime"]["inputs"]
            memo = generate_credit_memo(model, sample_inputs, "Jordan Taylor")
            memo_pdf_bytes = generate_underwriting_memo_pdf(memo, sample_inputs)
            st.download_button(
                label="📄 Download Underwriting Memo (PDF)",
                data=memo_pdf_bytes,
                file_name="FinRisk_Underwriting_Memo_Jordan_Taylor.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


# -----------------------------------------------------------------------------
# Page 8: Architecture & Methodology
# -----------------------------------------------------------------------------
def page_about() -> None:
    st.title("ℹ️ About FinRisk Lab Architecture")
    st.markdown(
        "**FinRisk Lab** trains credit-risk models on public LendingClub loans and validates them "
        "the way a bank's model-risk management team would: calibration, out-of-time stability, "
        "leakage, explainability, fairness proxies, and ongoing portfolio monitoring.\n\n"
        "- **Data:** LendingClub accepted loans 2007-2018Q4 (Kaggle), 36-month loans only.\n"
        "- **Split by time:** train 2012-13, validation 2014, out-of-time test 2015, monitoring 2016.\n"
        "- **Models:** WoE logistic-regression scorecard (champion) and XGBoost / LightGBM (challengers).\n"
        "- **Scoring never uses an LLM.** GenAI is used exclusively for narrative drafting behind a Number Guard.\n"
    )
    st.subheader("Master Scale Risk Bands")
    st.dataframe(band_table(), hide_index=True, use_container_width=True)

    model = load_model()
    if model is not None:
        st.subheader("Champion Scorecard Points")
        pts = pd.DataFrame(model.spec["points_table"])
        feat = st.selectbox("Feature", model.features)
        view = pts[pts["feature"] == feat][["bin", "n", "bad_rate", "woe", "points"]]
        st.dataframe(view.style.format({"bad_rate": "{:.1%}", "woe": "{:+.3f}", "points": "{:+.1f}"}),
                     hide_index=True, use_container_width=True)


# -----------------------------------------------------------------------------
# Main Application Router
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="FinRisk Lab | Credit Intelligence Platform", page_icon="🏦", layout="wide")
    apply_custom_styles()

    st.sidebar.title("🏦 FinRisk Lab")
    st.sidebar.caption("Credit-Risk Intelligence & Governance")

    # Sidebar Groq Key & Mode configuration
    with st.sidebar.expander("🔑 LLM / Groq Key Settings", expanded=False):
        api_key_input = st.text_input(
            "Groq API Key",
            type="password",
            value=os.environ.get("GROQ_API_KEY", "") if os.environ.get("GROQ_API_KEY") != "your_key_here" else "",
            help="Free key from console.groq.com. Leave empty to run the offline reasoning engine."
        )
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input.strip()
        groq_model = st.selectbox("LLM Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
        os.environ["GROQ_MODEL"] = groq_model

        active_llm = get_active_llm()
        if active_llm is not None:
            st.success("🟢 Live Groq LLM Connected")
        else:
            st.info("🧠 Offline Reasoning Engine Active")

    page = st.sidebar.radio("Navigation", PAGES)
    st.sidebar.divider()
    st.sidebar.caption(DISCLAIMER)

    router = {
        "🧑‍💼 Agentic Credit Analyst": page_agentic_analyst,
        "📊 Applicant Scoring & Sensitivity": page_score,
        "🛡️ Model Validation (SR 11-7)": page_validation,
        "🔬 Explanation Evaluation": page_explanation_evaluation,
        "📈 Monitoring & Stress Testing": page_monitoring,
        "🤖 Guarded GenAI & Regulatory RAG": page_genai_rag,
        "📄 Executive PDF Reports": page_pdf_reports,
        "ℹ️ Architecture & Methodology": page_about,
    }
    router[page]()


if __name__ == "__main__":
    main()
