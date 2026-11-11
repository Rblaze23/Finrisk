"""Time-based splits: the single source of truth for train / valid / oot / monitor.

Rules (years come from .env via config.py):
  - only loans with the configured term (36 months by default), so outcomes are mature
  - split by loan ISSUE YEAR, never at random
  - everything else is labelled 'unused'

Creates:
  staging.loan_splits   (loan_id, split)
  staging.model_sample  (view: loans_clean + split, without 'unused')

Usage:
    python -m finrisk.data.splits

In other modules:
    from finrisk.data.splits import load_split
    train = load_split("train")
"""
from __future__ import annotations

import re

import pandas as pd
from sqlalchemy import text

from finrisk.config import get_settings
from finrisk.db import connect, execute_sql, get_engine

SPLIT_ORDER = ["train", "valid", "oot", "monitor"]


def assign_splits() -> None:
    s = get_settings()
    cond = f"term_months = {s.loan_term_months} AND issue_year"
    sql = f"""
    DROP TABLE IF EXISTS staging.loan_splits CASCADE;

    CREATE TABLE staging.loan_splits AS
    SELECT
        loan_id,
        CASE
            WHEN {cond} BETWEEN {s.train_start_year} AND {s.train_end_year} THEN 'train'
            WHEN {cond} = {s.valid_year}   THEN 'valid'
            WHEN {cond} = {s.oot_year}     THEN 'oot'
            WHEN {cond} = {s.monitor_year} THEN 'monitor'
            ELSE 'unused'
        END AS split
    FROM staging.loans_clean;

    ALTER TABLE staging.loan_splits ADD PRIMARY KEY (loan_id);
    CREATE INDEX idx_loan_splits_split ON staging.loan_splits (split);

    CREATE OR REPLACE VIEW staging.model_sample AS
    SELECT c.*, s.split
    FROM staging.loans_clean c
    JOIN staging.loan_splits s USING (loan_id)
    WHERE s.split <> 'unused';
    """
    execute_sql(sql)


def load_split(split: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load one split from staging.model_sample as a DataFrame."""
    if split not in SPLIT_ORDER:
        raise ValueError(f"split must be one of {SPLIT_ORDER}")
    if columns:
        for c in columns:
            if not re.fullmatch(r"[a-z_0-9]+", c):
                raise ValueError(f"invalid column name: {c}")
        cols = ", ".join(columns)
    else:
        cols = "*"
    query = text(f"SELECT {cols} FROM staging.model_sample WHERE split = :split")
    return pd.read_sql(query, get_engine(), params={"split": split})


def summarize() -> pd.DataFrame:
    return pd.read_sql(
        text("""SELECT split,
                       COUNT(*)                                AS n_loans,
                       SUM(is_default)                         AS n_defaults,
                       ROUND(AVG(is_default)::numeric, 4)      AS default_rate,
                       MIN(issue_d)                            AS first_issue,
                       MAX(issue_d)                            AS last_issue
                FROM staging.model_sample GROUP BY split"""),
        get_engine(),
    )


def main() -> None:
    s = get_settings()
    print(f"Assigning splits: term={s.loan_term_months}m | train {s.train_start_year}-{s.train_end_year}"
          f" | valid {s.valid_year} | oot {s.oot_year} | monitor {s.monitor_year}")
    connect().close()  # friendly error if Docker is down
    assign_splits()

    df = summarize()
    df["order"] = df["split"].map({n: i for i, n in enumerate(SPLIT_ORDER)})
    df = df.sort_values("order").drop(columns="order").reset_index(drop=True)
    print()
    print(df.to_string(index=False))

    # sanity check: splits must be strictly ordered in time
    present = df["split"].tolist()
    missing = [x for x in SPLIT_ORDER if x not in present]
    if missing:
        print(f"\nWARNING: no loans in split(s): {missing}. "
              "Normal with a small --sample, otherwise check the years in .env.")
    for a, b in zip(df["split"][:-1], df["split"][1:]):
        last_a = df.loc[df["split"] == a, "last_issue"].iloc[0]
        first_b = df.loc[df["split"] == b, "first_issue"].iloc[0]
        if not last_a < first_b:
            raise SystemExit(f"Temporal leakage: {a} ends {last_a} but {b} starts {first_b}")
    print("\nOK: splits are strictly ordered in time.")


if __name__ == "__main__":
    main()
