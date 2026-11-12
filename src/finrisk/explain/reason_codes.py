"""Turn feature contributions into adverse-action style reason codes."""
from __future__ import annotations

import numpy as np
import pandas as pd

REASON_LABELS = {
    "loan_amnt": "Loan amount",
    "annual_inc": "Annual income",
    "dti": "Debt-to-income ratio",
    "emp_length_years": "Employment length (years)",
    "fico_mid": "Credit score (FICO)",
    "delinq_2yrs": "Delinquencies in last 2 years",
    "inq_last_6mths": "Credit inquiries in last 6 months",
    "mths_since_last_delinq": "Months since last delinquency",
    "mths_since_last_record": "Months since last public record",
    "open_acc": "Open credit lines",
    "pub_rec": "Public derogatory records",
    "revol_bal": "Revolving balance",
    "revol_util": "Revolving credit utilization (%)",
    "total_acc": "Total credit lines",
    "mort_acc": "Mortgage accounts",
    "pub_rec_bankruptcies": "Public record bankruptcies",
    "tot_cur_bal": "Total current balance",
    "total_rev_hi_lim": "Total revolving credit limit",
    "avg_cur_bal": "Average current balance",
    "bc_util": "Bankcard utilization (%)",
    "num_bc_sats": "Satisfactory bankcard accounts",
    "credit_history_years": "Length of credit history (years)",
    "loan_to_income": "Loan amount relative to income",
    "revol_bal_to_income": "Revolving balance relative to income",
    "home_ownership": "Home ownership status",
    "verification_status": "Income verification status",
    "purpose": "Loan purpose",
    "addr_state": "State of residence",
    "initial_list_status": "Initial listing status",
    "grade": "Lender risk grade",
    "sub_grade": "Lender risk sub-grade",
    "int_rate": "Interest rate assigned by lender",
}


def format_value(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "not available"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"
    return str(v)


def top_reasons(contributions: pd.Series, values: pd.Series, k: int = 5,
                min_impact: float = 0.01) -> list[dict]:
    """Top-k features that INCREASE the applicant's risk (log-odds contribution > min_impact)."""
    pos = contributions[contributions > min_impact].sort_values(ascending=False).head(k)
    return [
        {"feature": f, "label": REASON_LABELS.get(f, f),
         "value": format_value(values.get(f)), "impact": float(c)}
        for f, c in pos.items()
    ]


def reasons_text(reasons: list[dict]) -> list[str]:
    return [f"{i}. {r['label']}: {r['value']}  (+{r['impact']:.2f} log-odds)"
            for i, r in enumerate(reasons, 1)]
