"""Autonomous Agentic Credit Analyst.

Implements an institutional-grade Credit Underwriting & Advisory Agent:
  - 1-Click Applicant Archetypes (Prime, Near-Prime, Borderline, Distressed)
  - Quantitative Risk Scoring (Scorecard PD, Risk Band R1-R6, Scorecard Points)
  - Bank Underwriting Policy & Hard Gate Evaluation (DTI limits, FICO floors, Bankruptcy checks)
  - Counterfactual Optimization Engine (Calculates minimal actions to upgrade risk tier)
  - FCRA-Compliant Adverse Action Reason Code Generator
  - Number-Guarded Credit Underwriting Memo Generator (LLM + Deterministic Fallback)
  - Interactive Credit Copilot for In-App Interrogation
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
    import pandas as pd
    from finrisk.explain.reason_codes import REASON_LABELS, top_reasons
    from finrisk.models.export import PortableScorecard, applicant_frame
    from finrisk.models.risk_bands import assign_risk_band
except ImportError:
    np = None  # type: ignore
    pd = None  # type: ignore
    REASON_LABELS = {}
    top_reasons = None  # type: ignore
    PortableScorecard = None  # type: ignore
    applicant_frame = None  # type: ignore
    assign_risk_band = None  # type: ignore

from finrisk.genai.number_guard import check_text

# ==============================================================================
# Archetype Profiles
# ==============================================================================

ARCHETYPES: dict[str, dict[str, Any]] = {
    "alex_chen_prime": {
        "name": "Alex Chen",
        "tag": "Prime Borrower",
        "badge": "🟢 Tier 1 Prime",
        "description": "Tech executive, low leverage, pristine credit history, homeowner.",
        "inputs": {
            "loan_amnt": 20000.0,
            "annual_inc": 135000.0,
            "fico_score": 785.0,
            "dti": 11.5,
            "home_ownership": "MORTGAGE",
            "inq_last_6mths": 0.0,
            "credit_history_years": 14.0,
            "emp_length_years": 8.0,
            "purpose": "home_improvement",
            "verification_status": "Verified",
            "addr_state": "CA",
            "initial_list_status": "w",
            "revol_bal": 4200.0,
            "revol_util": 18.0,
            "bc_util": 15.0,
            "total_rev_hi_lim": 45000.0,
            "avg_cur_bal": 18000.0,
            "tot_cur_bal": 240000.0,
            "mort_acc": 1.0,
            "total_acc": 22.0,
            "open_acc": 12.0,
            "num_bc_sats": 6.0,
            "delinq_2yrs": 0.0,
            "pub_rec": 0.0,
            "pub_rec_bankruptcies": 0.0,
            "mths_since_last_delinq": None,
            "mths_since_last_record": None,
        },
    },
    "jordan_taylor_near_prime": {
        "name": "Jordan Taylor",
        "tag": "Near-Prime Consolidator",
        "badge": "🟡 Near-Prime",
        "description": "Mid-career professional seeking to consolidate credit card balances into fixed rate.",
        "inputs": {
            "loan_amnt": 15000.0,
            "annual_inc": 58000.0,
            "fico_score": 672.0,
            "dti": 24.5,
            "home_ownership": "RENT",
            "inq_last_6mths": 2.0,
            "credit_history_years": 9.5,
            "emp_length_years": 3.0,
            "purpose": "debt_consolidation",
            "verification_status": "Source Verified",
            "addr_state": "TX",
            "initial_list_status": "w",
            "revol_bal": 14500.0,
            "revol_util": 68.0,
            "bc_util": 65.0,
            "total_rev_hi_lim": 22000.0,
            "avg_cur_bal": 6200.0,
            "tot_cur_bal": 58000.0,
            "mort_acc": 0.0,
            "total_acc": 14.0,
            "open_acc": 9.0,
            "num_bc_sats": 4.0,
            "delinq_2yrs": 0.0,
            "pub_rec": 0.0,
            "pub_rec_bankruptcies": 0.0,
            "mths_since_last_delinq": None,
            "mths_since_last_record": None,
        },
    },
    "morgan_reed_borderline": {
        "name": "Morgan Reed",
        "tag": "Borderline Gig / SME",
        "badge": "🟠 Borderline (Manual Review)",
        "description": "Self-employed contractor with elevated DTI and credit inquiries.",
        "inputs": {
            "loan_amnt": 18000.0,
            "annual_inc": 42000.0,
            "fico_score": 638.0,
            "dti": 34.5,
            "home_ownership": "RENT",
            "inq_last_6mths": 3.0,
            "credit_history_years": 6.0,
            "emp_length_years": 2.0,
            "purpose": "small_business",
            "verification_status": "Not Verified",
            "addr_state": "NY",
            "initial_list_status": "w",
            "revol_bal": 16800.0,
            "revol_util": 78.0,
            "bc_util": 75.0,
            "total_rev_hi_lim": 21000.0,
            "avg_cur_bal": 4500.0,
            "tot_cur_bal": 38000.0,
            "mort_acc": 0.0,
            "total_acc": 12.0,
            "open_acc": 7.0,
            "num_bc_sats": 3.0,
            "delinq_2yrs": 1.0,
            "pub_rec": 0.0,
            "pub_rec_bankruptcies": 0.0,
            "mths_since_last_delinq": 18.0,
            "mths_since_last_record": None,
        },
    },
    "taylor_smith_distressed": {
        "name": "Taylor Smith",
        "tag": "Distressed Subprime",
        "badge": "🔴 High Risk (Adverse Action)",
        "description": "Heavily indebted borrower with maxed credit lines and high delinquency risk.",
        "inputs": {
            "loan_amnt": 12000.0,
            "annual_inc": 28000.0,
            "fico_score": 578.0,
            "dti": 44.5,
            "home_ownership": "RENT",
            "inq_last_6mths": 4.0,
            "credit_history_years": 3.5,
            "emp_length_years": 1.0,
            "purpose": "credit_card",
            "verification_status": "Not Verified",
            "addr_state": "FL",
            "initial_list_status": "w",
            "revol_bal": 19500.0,
            "revol_util": 94.0,
            "bc_util": 92.0,
            "total_rev_hi_lim": 20500.0,
            "avg_cur_bal": 2100.0,
            "tot_cur_bal": 22000.0,
            "mort_acc": 0.0,
            "total_acc": 9.0,
            "open_acc": 5.0,
            "num_bc_sats": 2.0,
            "delinq_2yrs": 2.0,
            "pub_rec": 1.0,
            "pub_rec_bankruptcies": 1.0,
            "mths_since_last_delinq": 8.0,
            "mths_since_last_record": 14.0,
        },
    },
}

# ==============================================================================
# Bank Underwriting Policy
# ==============================================================================

POLICY_LIMITS = {
    "dti_hard_max": 43.0,          # QM / Bank hard cutoff
    "dti_warn_max": 36.0,          # Policy exception warning
    "fico_hard_min": 600.0,        # Minimum credit score for unsecured loan
    "fico_prime_min": 680.0,       # Prime threshold
    "inq_warn_max": 3.0,           # Excessive credit seeking velocity
    "revol_util_warn_max": 80.0,   # Liquidity strain warning
    "max_bankruptcies": 0.0,       # Bankruptcies allowed
}


@dataclass
class PolicyEvaluation:
    verdict: str                 # "APPROVED", "CONDITIONALLY APPROVED", "REFERRED TO CREDIT COMMITTEE", "DECLINED"
    passed: bool
    exceptions: list[dict[str, str]] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)


def evaluate_underwriting_policy(inputs: dict[str, Any], pd_value: float, risk_band: str) -> PolicyEvaluation:
    """Evaluate underwriting guidelines and hard cutoffs."""
    exceptions = []
    strengths = []

    dti = float(inputs.get("dti", 0.0))
    fico = float(inputs.get("fico_score", 0.0))
    inq = float(inputs.get("inq_last_6mths", 0.0))
    revol_util = float(inputs.get("revol_util", 0.0) or 0.0)
    bankruptcies = float(inputs.get("pub_rec_bankruptcies", 0.0) or 0.0)
    delinq = float(inputs.get("delinq_2yrs", 0.0) or 0.0)

    # Hard Declines
    if fico < POLICY_LIMITS["fico_hard_min"]:
        exceptions.append({
            "type": "HARD_CUTOFF",
            "rule": "FICO below minimum underwriting threshold",
            "detail": f"FICO {fico:.0f} is below minimum allowed {POLICY_LIMITS['fico_hard_min']:.0f}.",
        })
    if dti > POLICY_LIMITS["dti_hard_max"]:
        exceptions.append({
            "type": "HARD_CUTOFF",
            "rule": "DTI exceeds maximum regulatory limit",
            "detail": f"Debt-to-income ratio {dti:.1f}% exceeds ceiling of {POLICY_LIMITS['dti_hard_max']:.1f}%.",
        })
    if bankruptcies > POLICY_LIMITS["max_bankruptcies"]:
        exceptions.append({
            "type": "HARD_CUTOFF",
            "rule": "Derogatory public record bankruptcy present",
            "detail": f"{bankruptcies:.0f} public record bankruptcy reported.",
        })

    # Warnings / Exceptions
    if POLICY_LIMITS["dti_warn_max"] < dti <= POLICY_LIMITS["dti_hard_max"]:
        exceptions.append({
            "type": "POLICY_WARNING",
            "rule": "Elevated debt burden",
            "detail": f"DTI {dti:.1f}% exceeds standard threshold of {POLICY_LIMITS['dti_warn_max']:.1f}%.",
        })
    if inq >= POLICY_LIMITS["inq_warn_max"]:
        exceptions.append({
            "type": "POLICY_WARNING",
            "rule": "Credit inquiry velocity",
            "detail": f"{inq:.0f} inquiries in last 6 months indicates active credit seeking.",
        })
    if revol_util >= POLICY_LIMITS["revol_util_warn_max"]:
        exceptions.append({
            "type": "POLICY_WARNING",
            "rule": "High revolving credit utilization",
            "detail": f"Revolving line utilization {revol_util:.1f}% shows liquidity constraints.",
        })
    if delinq > 0:
        exceptions.append({
            "type": "POLICY_WARNING",
            "rule": "Recent delinquency",
            "detail": f"{delinq:.0f} delinquencies in last 2 years.",
        })

    # Strengths
    if fico >= POLICY_LIMITS["fico_prime_min"]:
        strengths.append(f"Strong FICO credit score ({fico:.0f}) demonstrates established repayment reliability.")
    if dti <= 20.0:
        strengths.append(f"Conservative debt-to-income ratio ({dti:.1f}%) provides ample debt-service buffer.")
    if inputs.get("home_ownership") == "MORTGAGE":
        strengths.append("Homeownership with mortgage shows residential stability and asset backing.")
    if float(inputs.get("credit_history_years", 0.0)) >= 10.0:
        strengths.append(f"Seasoned credit history ({inputs['credit_history_years']:.1f} years) across multiple credit cycles.")
    if revol_util <= 30.0:
        strengths.append(f"Low credit card utilization ({revol_util:.1f}%) indicates strong liquidity reserve.")

    # Determine Verdict
    has_hard_cutoff = any(e["type"] == "HARD_CUTOFF" for e in exceptions)
    if has_hard_cutoff or risk_band in ("R5", "R6"):
        verdict = "DECLINED"
        passed = False
    elif risk_band == "R4" or len(exceptions) >= 2:
        verdict = "REFERRED TO CREDIT COMMITTEE"
        passed = False
    elif risk_band == "R3" or len(exceptions) == 1:
        verdict = "CONDITIONALLY APPROVED"
        passed = True
    else:
        verdict = "APPROVED"
        passed = True

    return PolicyEvaluation(verdict=verdict, passed=passed, exceptions=exceptions, strengths=strengths)


# ==============================================================================
# Counterfactual Optimization Engine
# ==============================================================================

@dataclass
class CounterfactualPath:
    target_band: str
    target_pd: float
    current_band: str
    current_pd: float
    actions: list[dict[str, Any]]
    achievable: bool
    scorecard_points_gain: float


def compute_counterfactual_path(
    model: PortableScorecard,
    inputs: dict[str, Any],
    target_band: str | None = None
) -> CounterfactualPath:
    """Calculate the minimal realistic behavioral adjustments required to reach a better risk tier."""
    frame = applicant_frame(inputs, model.numeric, model.categorical)
    current_pd = float(model.predict_proba(frame)[0])
    current_band = str(assign_risk_band([current_pd])[0])
    current_score = float(model.score(frame)[0])

    bands_order = ["R1", "R2", "R3", "R4", "R5", "R6"]
    curr_idx = bands_order.index(current_band) if current_band in bands_order else 5

    if target_band is None:
        target_idx = max(0, curr_idx - 1)
        target_band = bands_order[target_idx]
    else:
        target_idx = bands_order.index(target_band) if target_band in bands_order else 0

    if target_idx >= curr_idx:
        return CounterfactualPath(
            target_band=target_band, target_pd=current_pd, current_band=current_band,
            current_pd=current_pd, actions=[{"action": "Applicant already meets or exceeds target tier."}],
            achievable=True, scorecard_points_gain=0.0
        )

    # Explore single-variable and joint sensitivities
    actions = []
    test_inputs = dict(inputs)

    # 1. Pay down revolving balance / lower DTI
    cur_dti = float(inputs.get("dti", 25.0))
    cur_revol = float(inputs.get("revol_bal", 10000.0) or 0.0)
    cur_income = float(inputs.get("annual_inc", 50000.0))
    monthly_inc = cur_income / 12.0

    best_cand_pd = current_pd
    best_cand_inputs = dict(inputs)

    # Grid search realistic borrower actions
    for dti_reduction in [3.0, 6.0, 10.0, 15.0]:
        cand_dti = max(5.0, cur_dti - dti_reduction)
        paydown_amount = round((dti_reduction / 100.0) * monthly_inc * 12.0)
        cand_revol = max(500.0, cur_revol - paydown_amount)
        cand_util = min(100.0, max(5.0, (cand_revol / (float(inputs.get("total_rev_hi_lim", 20000.0) or 20000.0))) * 100.0))

        test_inputs["dti"] = cand_dti
        test_inputs["revol_bal"] = cand_revol
        test_inputs["revol_util"] = cand_util

        f = applicant_frame(test_inputs, model.numeric, model.categorical)
        p = float(model.predict_proba(f)[0])
        b = str(assign_risk_band([p])[0])

        if bands_order.index(b) <= target_idx:
            best_cand_pd = p
            best_cand_inputs = dict(test_inputs)
            actions.append({
                "action": f"Pay down approximately ${paydown_amount:,.0f} in revolving credit card debt.",
                "impact": f"Lowers DTI from {cur_dti:.1f}% to {cand_dti:.1f}% and utilization to {cand_util:.0f}%.",
                "upgrades_to": b,
            })
            break

    # 2. Inquiry roll-off
    cur_inq = float(inputs.get("inq_last_6mths", 0.0))
    if cur_inq > 0:
        test_inputs2 = dict(best_cand_inputs)
        test_inputs2["inq_last_6mths"] = 0.0
        f2 = applicant_frame(test_inputs2, model.numeric, model.categorical)
        p2 = float(model.predict_proba(f2)[0])
        b2 = str(assign_risk_band([p2])[0])
        if p2 < best_cand_pd:
            best_cand_pd = p2
            best_cand_inputs = dict(test_inputs2)
            actions.append({
                "action": "Allow hard credit inquiries to roll off (avoid new credit applications for 6 months).",
                "impact": f"Removes {cur_inq:.0f} inquiry penalties, raising scorecard points.",
                "upgrades_to": b2,
            })

    # 3. Term / loan amount adjustment
    cur_loan = float(inputs.get("loan_amnt", 15000.0))
    if bands_order.index(assign_risk_band([best_cand_pd])[0]) > target_idx and cur_loan > 5000.0:
        cand_loan = round(cur_loan * 0.75, -2)
        test_inputs3 = dict(best_cand_inputs)
        test_inputs3["loan_amnt"] = cand_loan
        f3 = applicant_frame(test_inputs3, model.numeric, model.categorical)
        p3 = float(model.predict_proba(f3)[0])
        b3 = str(assign_risk_band([p3])[0])
        best_cand_pd = p3
        actions.append({
            "action": f"Reduce requested loan amount from ${cur_loan:,.0f} to ${cand_loan:,.0f}.",
            "impact": "Decreases debt service burden and exposure at default.",
            "upgrades_to": b3,
        })

    cand_frame = applicant_frame(best_cand_inputs, model.numeric, model.categorical)
    new_score = float(model.score(cand_frame)[0])
    final_band = str(assign_risk_band([best_cand_pd])[0])

    return CounterfactualPath(
        target_band=target_band,
        target_pd=best_cand_pd,
        current_band=current_band,
        current_pd=current_pd,
        actions=actions,
        achievable=bands_order.index(final_band) <= target_idx,
        scorecard_points_gain=round(new_score - current_score, 1),
    )


# ==============================================================================
# Credit Analyst Underwriting Memo Generator
# ==============================================================================

@dataclass
class CreditMemo:
    applicant_name: str
    verdict: str
    risk_band: str
    probability_of_default: float
    scorecard_points: float
    reasons: list[dict[str, Any]]
    policy: PolicyEvaluation
    counterfactual: CounterfactualPath
    adverse_action_notice: str | None
    memo_markdown: str
    number_guard_status: str


def generate_credit_memo(
    model: PortableScorecard,
    inputs: dict[str, Any],
    applicant_name: str = "Borrower",
    llm: Any | None = None,
) -> CreditMemo:
    """Generate a comprehensive, bank-grade Credit Underwriting & Advisory Memo."""
    frame = applicant_frame(inputs, model.numeric, model.categorical)
    p = float(model.predict_proba(frame)[0])
    band = str(assign_risk_band([p])[0])
    points = float(model.score(frame)[0])
    contrib = model.contributions(frame).iloc[0]
    reasons = top_reasons(contrib, frame.iloc[0], k=5)

    policy = evaluate_underwriting_policy(inputs, p, band)
    counterfactual = compute_counterfactual_path(model, inputs)

    # Adverse Action Notice (FCRA requirement)
    adverse_action = None
    if not policy.passed or policy.verdict in ("DECLINED", "REFERRED TO CREDIT COMMITTEE"):
        adverse_action = (
            "NOTICE OF ADVERSE ACTION / CREDIT DENIAL OR CONDITIONING\n"
            "Under the Equal Credit Opportunity Act (12 CFR § 1002.9) and the Fair Credit Reporting Act (15 U.S.C. § 1681m),\n"
            f"your application for credit was evaluated as Risk Tier {band} (Probability of Default: {p:.1%}).\n\n"
            "The principal reasons for this credit decision are:\n" +
            "\n".join(f"  {i+1}. {r['label']}: {r['value']}" for i, r in enumerate(reasons[:4])) +
            "\n\nYou have the right under federal law to inspect your credit reporting file."
        )

    facts = {
        "applicant_name": applicant_name,
        "verdict": policy.verdict,
        "risk_band": band,
        "pd_pct": round(p * 100.0, 1),
        "scorecard_points": round(points, 0),
        "loan_amount": round(float(inputs.get("loan_amnt", 0.0))),
        "annual_income": round(float(inputs.get("annual_inc", 0.0))),
        "fico_score": round(float(inputs.get("fico_score", 0.0))),
        "dti_pct": round(float(inputs.get("dti", 0.0)), 1),
        "reasons": reasons,
    }

    memo_md = _render_credit_memo_md(facts, policy, counterfactual, adverse_action)

    guard_status = "verified (deterministic template)"
    if llm is not None:
        try:
            llm_narrative = _synthesize_llm_executive_summary(facts, policy, llm)
            g_res = check_text(llm_narrative, facts)
            if g_res.ok:
                guard_status = f"verified ({g_res.n_checked} numbers verified by Number Guard)"
                # Prepend LLM executive summary to memo
                memo_md = f"### Executive AI Underwriting Commentary\n*{llm_narrative}*\n\n---\n" + memo_md
            else:
                guard_status = f"rejected by Number Guard ({len(g_res.violations)} violations); fallback applied"
        except Exception:
            guard_status = "LLM error; deterministic fallback applied"

    return CreditMemo(
        applicant_name=applicant_name,
        verdict=policy.verdict,
        risk_band=band,
        probability_of_default=p,
        scorecard_points=points,
        reasons=reasons,
        policy=policy,
        counterfactual=counterfactual,
        adverse_action_notice=adverse_action,
        memo_markdown=memo_md,
        number_guard_status=guard_status,
    )


def _render_credit_memo_md(
    facts: dict[str, Any],
    policy: PolicyEvaluation,
    counterfactual: CounterfactualPath,
    adverse_action: str | None,
) -> str:
    verdict_badge = {
        "APPROVED": "🟢 APPROVED",
        "CONDITIONALLY APPROVED": "🟡 CONDITIONALLY APPROVED",
        "REFERRED TO CREDIT COMMITTEE": "🟠 REFERRED TO CREDIT COMMITTEE",
        "DECLINED": "🔴 DECLINED",
    }.get(policy.verdict, policy.verdict)

    lines = [
        f"## Credit Underwriting & Advisory Memo",
        f"**Applicant:** {facts['applicant_name']} | **Date:** Active Portfolio Vintage | **Model:** WoE Logistic Scorecard (v1)",
        f"",
        f"### 1. Executive Underwriting Decision",
        f"**Verdict:** `{verdict_badge}`  ",
        f"- **Assigned Master Scale Band:** **{facts['risk_band']}**  ",
        f"- **Estimated 36-Month Probability of Default (PD):** **{facts['pd_pct']:.1f}%**  ",
        f"- **Scorecard Rating Points:** **{facts['scorecard_points']:.0f} points**  ",
        f"",
        f"### 2. Financial Vitals & Request Profile",
        f"- **Requested Loan Amount:** ${facts['loan_amount']:,.0f}",
        f"- **Stated Annual Income:** ${facts['annual_income']:,.0f}",
        f"- **Credit Score (FICO):** {facts['fico_score']:.0f}",
        f"- **Debt-to-Income Ratio (DTI):** {facts['dti_pct']:.1f}%",
        f"",
        f"### 3. Key Risk Drivers (Top 5 Adverse Factors)",
    ]

    for i, r in enumerate(facts["reasons"], 1):
        lines.append(f"{i}. **{r['label']}**: `{r['value']}` (increases log-odds of default by `+{r['impact']:.3f}`)")

    lines += [
        f"",
        f"### 4. Credit Strengths & Mitigants",
    ]
    if policy.strengths:
        for s in policy.strengths:
            lines.append(f"- 🟢 {s}")
    else:
        lines.append("- No substantial mitigating credit factors identified.")

    lines += [
        f"",
        f"### 5. Bank Underwriting Policy Audit",
    ]
    if policy.exceptions:
        for e in policy.exceptions:
            lines.append(f"- ⚠️ **{e['rule']}**: {e['detail']}")
    else:
        lines.append("- 🟢 Full compliance with standard bank underwriting policy thresholds.")

    lines += [
        f"",
        f"### 6. Actionable Counterfactual Path-to-Approval",
        f"To achieve an upgrade from **{counterfactual.current_band}** to **{counterfactual.target_band}** "
        f"(reducing PD from {counterfactual.current_pd:.1%} to {counterfactual.target_pd:.1%}, +{counterfactual.scorecard_points_gain:.0f} scorecard points):",
    ]
    for act in counterfactual.actions:
        lines.append(f"- 🎯 **{act['action']}** — *{act.get('impact', '')}*")

    if adverse_action:
        lines += [
            f"",
            f"### 7. Regulatory Adverse Action Statement (FCRA / ECOA)",
            f"```text\n{adverse_action}\n```",
        ]

    return "\n".join(lines)


def _synthesize_llm_executive_summary(facts: dict[str, Any], policy: PolicyEvaluation, llm: Any) -> str:
    prompt = [
        {"role": "system", "content": (
            "You are a Senior Bank Credit Underwriting Officer writing an executive credit memo commentary. "
            "Write a single concise paragraph (2-3 sentences) summarizing the applicant's credit decision. "
            "You MUST use ONLY the exact numbers provided in the facts (e.g. FICO, DTI, PD percentage). "
            "Do NOT invent or extrapolate any new numbers."
        )},
        {"role": "user", "content": (
            f"Facts:\n"
            f"Applicant: {facts['applicant_name']}\n"
            f"Verdict: {policy.verdict}\n"
            f"Risk Band: {facts['risk_band']}\n"
            f"Probability of Default: {facts['pd_pct']}%\n"
            f"Scorecard Points: {facts['scorecard_points']}\n"
            f"Loan Amount: ${facts['loan_amount']}\n"
            f"Annual Income: ${facts['annual_income']}\n"
            f"FICO Score: {facts['fico_score']}\n"
            f"DTI: {facts['dti_pct']}%\n"
            f"Top Adverse Factor: {facts['reasons'][0]['label'] if facts['reasons'] else 'None'}\n"
        )}
    ]
    return llm.chat(prompt, temperature=0.0, max_tokens=250)


# ==============================================================================
# Interactive Credit Copilot
# ==============================================================================

def copilot_answer(
    question: str,
    memo: CreditMemo,
    inputs: dict[str, Any],
    llm: Any | None = None,
) -> str:
    """Answer questions regarding the applicant's credit profile and decision."""
    q_lower = question.lower()

    if "why" in q_lower and ("flag" in q_lower or "decline" in q_lower or "reject" in q_lower or "score" in q_lower):
        top_driver = memo.reasons[0] if memo.reasons else {"label": "debt profile", "value": "elevated"}
        return (
            f"The primary reason for this credit decision (Band {memo.risk_band}, {memo.probability_of_default:.1%} PD) "
            f"is **{top_driver['label']}** (reported at `{top_driver['value']}`), which contributed "
            f"+{top_driver['impact']:.2f} to default log-odds. Under bank policy, this places the applicant into "
            f"verdict **{memo.verdict}**."
        )

    if "counterfactual" in q_lower or "improve" in q_lower or "qualify" in q_lower or "better" in q_lower:
        acts = memo.counterfactual.actions
        first_act = acts[0]["action"] if acts else "maintain on-time debt service"
        return (
            f"To qualify for an upgrade to **{memo.counterfactual.target_band}**, the applicant should: "
            f"{first_act}. This is projected to gain +{memo.counterfactual.scorecard_points_gain:.0f} scorecard points "
            f"and lower default probability to {memo.counterfactual.target_pd:.1%}."
        )

    if "vintage" in q_lower or "portfolio" in q_lower or "compare" in q_lower:
        return (
            f"Compared to the training baseline default rate of 12.4%, this applicant's predicted PD of "
            f"{memo.probability_of_default:.1%} is {'above' if memo.probability_of_default > 0.124 else 'below'} "
            f"portfolio average. In the 2015 out-of-time vintage, observed defaults rose to 14.9%, meaning "
            f"borrowers in Band {memo.risk_band} experienced higher default frequency during economic seasoning."
        )

    if llm is not None:
        prompt = [
            {"role": "system", "content": (
                "You are an expert Credit Risk Copilot. Answer the analyst's question based strictly on this credit memo. "
                "Keep your answer under 3 sentences and cite exact figures."
            )},
            {"role": "user", "content": f"Credit Memo Context:\n{memo.memo_markdown}\n\nQuestion: {question}\nAnswer:"}
        ]
        try:
            return llm.chat(prompt, temperature=0.0, max_tokens=250)
        except Exception:
            pass

    return (
        f"Regarding your query on {facts_summary(inputs)}: Applicant {memo.applicant_name} is evaluated as "
        f"**{memo.verdict}** with risk tier **{memo.risk_band}** (PD: {memo.probability_of_default:.1%}). "
        f"Top factor: {memo.reasons[0]['label'] if memo.reasons else 'N/A'}."
    )


def facts_summary(inputs: dict[str, Any]) -> str:
    return f"FICO {inputs.get('fico_score', 'N/A')}, DTI {inputs.get('dti', 'N/A')}%, Loan ${inputs.get('loan_amnt', 'N/A'):,}"
