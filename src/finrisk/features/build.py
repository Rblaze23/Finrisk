"""Feature table, feature sets and loaders.

Usage:
    python -m finrisk.features.build        # build features.loan_features + missingness report

In other modules:
    from finrisk.features.build import load_features, get_feature_columns
    train = load_features("train")
    numeric, categorical = get_feature_columns("independent")
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from finrisk.config import get_settings
from finrisk.data.columns import LENDER_ASSIGNED_COLUMNS
from finrisk.db import connect, execute_sql, get_engine

pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 100)

NUMERIC_FEATURES = [
    "loan_amnt", "annual_inc", "dti", "emp_length_years", "fico_mid",
    "delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq",
    "mths_since_last_record", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "mort_acc", "pub_rec_bankruptcies", "tot_cur_bal",
    "total_rev_hi_lim", "avg_cur_bal", "bc_util", "num_bc_sats",
    "credit_history_years", "loan_to_income", "revol_bal_to_income",
]
CATEGORICAL_FEATURES = [
    "home_ownership", "verification_status", "purpose", "addr_state",
    "initial_list_status",
]
LENDER_NUMERIC = ["int_rate"]
LENDER_CATEGORICAL = ["grade", "sub_grade"]

assert set(LENDER_NUMERIC + LENDER_CATEGORICAL) == set(LENDER_ASSIGNED_COLUMNS)

# independent  : what a bank with its own data would have
# with_lender  : adds LendingClub's own grade / sub_grade / int_rate
FEATURE_SETS = {
    "independent": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    "with_lender": (NUMERIC_FEATURES + LENDER_NUMERIC,
                    CATEGORICAL_FEATURES + LENDER_CATEGORICAL),
}


def get_feature_columns(feature_set: str) -> tuple[list[str], list[str]]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {list(FEATURE_SETS)}")
    num, cat = FEATURE_SETS[feature_set]
    return list(num), list(cat)


def build_features() -> None:
    sql = (get_settings().sql_dir / "03_features.sql").read_text(encoding="utf-8")
    execute_sql(sql)


def load_features(split: str | None = None) -> pd.DataFrame:
    """Load features.loan_features (optionally one split) with numeric dtypes enforced."""
    query = "SELECT * FROM features.loan_features"
    params = {}
    if split is not None:
        query += " WHERE split = :split"
        params = {"split": split}
    df = pd.read_sql(text(query), get_engine(), params=params)
    numeric = set(NUMERIC_FEATURES + LENDER_NUMERIC)
    for c in df.columns:
        if c in numeric:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    connect().close()  # friendly error if Docker is down
    print("Building features.loan_features ...")
    build_features()

    df = load_features()
    print(f"\n{len(df):,} loans\n")
    print(df.groupby("split")["is_default"].agg(n_loans="size", default_rate="mean")
          .reindex(["train", "valid", "oot", "monitor"]).round(4).to_string())

    feats = NUMERIC_FEATURES + CATEGORICAL_FEATURES + LENDER_NUMERIC + LENDER_CATEGORICAL
    miss = df[feats].isna().groupby(df["issue_year"]).mean().T * 100
    miss = miss[miss.max(axis=1) > 5].round(1)
    print("\nMissing values (%) by issue year, for features with more than 5% missing anywhere:")
    print(miss.to_string())
    print("\nNext: python -m finrisk.models.scorecard")


if __name__ == "__main__":
    main()
