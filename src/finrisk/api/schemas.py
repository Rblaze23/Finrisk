"""Request and response schemas for the scoring API."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from finrisk.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES

DISCLAIMER = ("Demonstration model trained on historical LendingClub loans (2012-2013). "
              "Not for real credit decisions.")

# numeric features the caller does NOT provide directly (derived or renamed)
DERIVED = {"fico_mid", "loan_to_income", "revol_bal_to_income"}


class Applicant(BaseModel):
    """One loan application. Only the first three fields are required; any field you omit is
    treated as missing, exactly as in the training data."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "loan_amnt": 15000, "annual_inc": 52000, "fico_score": 672, "dti": 24.5,
        "emp_length_years": 3, "home_ownership": "RENT", "verification_status": "Verified",
        "purpose": "debt_consolidation", "addr_state": "CA", "inq_last_6mths": 2,
        "revol_bal": 14500, "revol_util": 71.0, "total_acc": 14, "open_acc": 9,
        "credit_history_years": 9.5, "initial_list_status": "w"}})

    loan_amnt: float = Field(..., gt=0, description="Requested loan amount")
    annual_inc: float = Field(..., gt=0, description="Annual income")
    fico_score: float = Field(..., ge=300, le=850, description="FICO score at application")

    dti: float | None = Field(None, ge=0, le=1000, description="Debt-to-income ratio (%)")
    emp_length_years: float | None = Field(None, ge=0, le=60)
    delinq_2yrs: float | None = Field(None, ge=0)
    inq_last_6mths: float | None = Field(None, ge=0)
    mths_since_last_delinq: float | None = Field(None, ge=0)
    mths_since_last_record: float | None = Field(None, ge=0)
    open_acc: float | None = Field(None, ge=0)
    pub_rec: float | None = Field(None, ge=0)
    revol_bal: float | None = Field(None, ge=0)
    revol_util: float | None = Field(None, ge=0, le=1000, description="Revolving utilization (%)")
    total_acc: float | None = Field(None, ge=0)
    mort_acc: float | None = Field(None, ge=0)
    pub_rec_bankruptcies: float | None = Field(None, ge=0)
    tot_cur_bal: float | None = Field(None, ge=0)
    total_rev_hi_lim: float | None = Field(None, ge=0)
    avg_cur_bal: float | None = Field(None, ge=0)
    bc_util: float | None = Field(None, ge=0, le=1000)
    num_bc_sats: float | None = Field(None, ge=0)
    credit_history_years: float | None = Field(None, ge=0, le=100)

    home_ownership: str | None = Field(None, description="RENT | MORTGAGE | OWN | OTHER")
    verification_status: str | None = None
    purpose: str | None = None
    addr_state: str | None = Field(None, description="US state code, e.g. CA")
    initial_list_status: str | None = None

    def to_frame(self) -> pd.DataFrame:
        """One-row frame with exactly the model's feature columns (derived ones computed)."""
        d = self.model_dump()
        row = {c: np.nan for c in NUMERIC_FEATURES}
        for k, v in d.items():
            if k in NUMERIC_FEATURES and v is not None:
                row[k] = float(v)
        row["fico_mid"] = float(d["fico_score"])
        row["loan_to_income"] = d["loan_amnt"] / d["annual_inc"]
        if d.get("revol_bal") is not None:
            row["revol_bal_to_income"] = d["revol_bal"] / d["annual_inc"]
        for c in CATEGORICAL_FEATURES:
            row[c] = d.get(c)
        return pd.DataFrame([row])


class Reason(BaseModel):
    feature: str
    label: str
    value: str
    impact: float = Field(..., description="Contribution to the log-odds of default (positive = raises risk)")


class ModelScore(BaseModel):
    role: str = Field(..., description="champion | challenger")
    kind: str = Field(..., description="scorecard | xgboost | lightgbm")
    version: str
    pd: float = Field(..., description="Estimated probability of default")
    risk_band: str = Field(..., description="Master-scale band R1 (safest) .. R6")
    reasons: list[Reason] = Field(..., description="Top factors that raise this applicant's risk")


class ScoreResponse(BaseModel):
    champion: ModelScore
    challenger: ModelScore | None = None
    disclaimer: str = DISCLAIMER
