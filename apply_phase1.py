#!/usr/bin/env python3
"""
Phase 1 for FinRisk Lab: data ingestion, cleaning, target and time splits.

Run from the ROOT of your repo (after apply_phase0.py):

    python apply_phase1.py            # fills the stub files
    python apply_phase1.py --force    # overwrite even if you edited them

Files that you already edited (more than a few lines) are skipped unless --force.
"""
import sys
from pathlib import Path

ROOT = Path(".").resolve()
FORCE = "--force" in sys.argv

FILES = {}

# ---------------------------------------------------------------------------
FILES["src/finrisk/db.py"] = r'''"""Database helpers: psycopg2 connection (bulk load / DDL) and SQLAlchemy engine (pandas)."""
from __future__ import annotations

import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from finrisk.config import get_settings


def connect():
    """Open a psycopg2 connection to the finrisk database."""
    s = get_settings()
    try:
        return psycopg2.connect(
            host=s.postgres_host,
            port=s.postgres_port,
            user=s.postgres_user,
            password=s.postgres_password,
            dbname=s.postgres_db,
        )
    except psycopg2.OperationalError as exc:
        raise SystemExit(
            f"Cannot connect to PostgreSQL at {s.postgres_host}:{s.postgres_port}.\n"
            "Is Docker running? Try: docker compose up -d\n"
            f"Original error: {exc}"
        )


def get_engine() -> Engine:
    """SQLAlchemy engine, used for pandas.read_sql."""
    return create_engine(get_settings().database_url)


def execute_sql(sql: str) -> None:
    """Run a (possibly multi-statement) SQL script in one transaction."""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/data/columns.py"] = r'''"""Column catalogue for the LendingClub accepted-loans file.

Single source of truth for which columns are loaded and what they mean for
modelling. The leakage tests (validation/leakage.py) use POST_ORIGINATION_COLUMNS.
"""

ID_COLUMN = "id"
STATUS_COLUMN = "loan_status"

# --- Known at application / origination time --------------------------------
NUMERIC_ORIGINATION = [
    "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc", "dti",
    "delinq_2yrs", "fico_range_low", "fico_range_high", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",
    "collections_12_mths_ex_med", "acc_now_delinq", "tot_coll_amt",
    "tot_cur_bal", "total_rev_hi_lim", "avg_cur_bal", "bc_util", "num_bc_sats",
]

CATEGORICAL_ORIGINATION = [
    "term", "grade", "sub_grade", "emp_length", "home_ownership",
    "verification_status", "purpose", "zip_code", "addr_state",
    "initial_list_status", "application_type",
]

DATE_ORIGINATION = ["issue_d", "earliest_cr_line"]

ORIGINATION_COLUMNS = NUMERIC_ORIGINATION + CATEGORICAL_ORIGINATION + DATE_ORIGINATION

# Outputs of LendingClub's own underwriting model. A bank building an
# independent PD model would not have these, so Phase 2 trains with and without.
LENDER_ASSIGNED_COLUMNS = ["grade", "sub_grade", "int_rate"]

# --- Only known AFTER the loan was issued (leakage candidates) --------------
POST_ORIGINATION_COLUMNS = [
    "out_prncp", "total_pymnt", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low", "debt_settlement_flag",
]

# Everything loaded into raw.loans (order preserved, no duplicates)
RAW_COLUMNS = list(
    dict.fromkeys(
        [ID_COLUMN, STATUS_COLUMN] + ORIGINATION_COLUMNS + POST_ORIGINATION_COLUMNS
    )
)
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/data/ingest.py"] = r'''"""Load the LendingClub accepted-loans CSV into PostgreSQL (raw.loans).

Every column is stored as TEXT, untouched. Type casting happens in SQL
(sql/02_staging.sql), so the raw layer stays a faithful copy of the source.

Usage:
    python -m finrisk.data.ingest --sample        # random 10% (fast, for development)
    python -m finrisk.data.ingest --sample 0.25   # random 25%
    python -m finrisk.data.ingest                 # full file (about 2.26M rows)
"""
from __future__ import annotations

import argparse
import io
import time

import pandas as pd

from finrisk.config import get_settings
from finrisk.data.columns import RAW_COLUMNS
from finrisk.db import connect


def check_columns(path, wanted: list[str]) -> list[str]:
    header = list(pd.read_csv(path, nrows=0, encoding_errors="replace").columns)
    missing = [c for c in wanted if c not in header]
    if missing:
        raise SystemExit(
            "These expected columns are not in the file:\n  "
            + ", ".join(missing)
            + "\nEdit src/finrisk/data/columns.py (and sql/02_staging.sql) to match."
        )
    return wanted


def create_raw_table(cur, columns: list[str]) -> None:
    cols_sql = ",\n    ".join(f'"{c}" TEXT' for c in columns)
    cur.execute("DROP TABLE IF EXISTS raw.loans CASCADE")
    cur.execute(f"CREATE TABLE raw.loans (\n    {cols_sql}\n)")


def iter_chunks(path, columns, chunksize, fraction, seed):
    reader = pd.read_csv(
        path,
        usecols=columns,
        dtype=str,
        chunksize=chunksize,
        encoding_errors="replace",
    )
    for chunk in reader:
        chunk = chunk[columns]
        # the file ends with summary text rows that are not loans
        chunk = chunk[chunk["id"].fillna("").str.fullmatch(r"[0-9]+")]
        if fraction < 1.0:
            chunk = chunk.sample(frac=fraction, random_state=seed)
        yield chunk


def copy_chunk(cur, chunk: pd.DataFrame, columns: list[str]) -> None:
    buf = io.StringIO()
    chunk.to_csv(buf, index=False, header=False)  # NaN -> empty -> NULL
    buf.seek(0)
    col_list = ", ".join(columns)
    cur.copy_expert(
        f"COPY raw.loans ({col_list}) FROM STDIN WITH (FORMAT csv, NULL '')", buf
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sample", nargs="?", const=0.1, type=float, default=None,
                        help="random fraction of rows to load (default 0.1 when flag is given)")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fraction = 1.0 if args.sample is None else args.sample
    if not 0 < fraction <= 1:
        raise SystemExit("--sample must be in (0, 1]")

    path = get_settings().raw_data_path
    if not path.exists():
        raise SystemExit(f"Raw data file not found: {path}")

    columns = check_columns(path, RAW_COLUMNS)
    print(f"Loading {path.name}  |  columns: {len(columns)}  |  fraction: {fraction:.0%}")

    start = time.time()
    total = 0
    conn = connect()
    try:
        with conn.cursor() as cur:
            create_raw_table(cur, columns)
            for i, chunk in enumerate(iter_chunks(path, columns, args.chunksize, fraction, args.seed), 1):
                copy_chunk(cur, chunk, columns)
                total += len(chunk)
                print(f"  chunk {i:>3}: {total:>9,} rows loaded  ({time.time() - start:5.0f}s)")
            cur.execute("ANALYZE raw.loans")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"Done: {total:,} rows in raw.loans ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
FILES["sql/02_staging.sql"] = r'''-- 02_staging.sql
-- raw.loans (all TEXT) -> staging.loans_clean (typed, with the default target).
--
-- Target definition:
--   is_default = 1 : loan_status contains 'Charged Off' or 'Default'
--   is_default = 0 : loan_status contains 'Fully Paid'
--   anything else (Current, Late, In Grace Period, ...) has no final outcome
--   yet and is EXCLUDED. This is documented in the validation report.
--
-- Re-running this script drops dependent objects (splits table and view).
-- Run `python -m finrisk.data.splits` afterwards.

-- ---------- safe parsing helpers ---------------------------------------------
CREATE OR REPLACE FUNCTION staging.to_num(t TEXT) RETURNS DOUBLE PRECISION
LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE WHEN c ~ '^-?[0-9]+([.][0-9]+)?$' THEN c::double precision END
    FROM (SELECT regexp_replace(t, '[^0-9.-]', '', 'g') AS c) s
$$;

-- 'Dec-2015' -> 2015-12-01
CREATE OR REPLACE FUNCTION staging.to_month(t TEXT) RETURNS DATE
LANGUAGE sql STABLE AS $$
    SELECT CASE WHEN t ~ '^[A-Za-z]{3}-[0-9]{4}$' THEN to_date(t, 'Mon-YYYY') END
$$;

-- ---------- staging table ------------------------------------------------------
DROP TABLE IF EXISTS staging.loans_clean CASCADE;

CREATE TABLE staging.loans_clean AS
WITH typed AS (
    SELECT
        r.id::bigint                                              AS loan_id,
        staging.to_month(r.issue_d)                               AS issue_d,
        EXTRACT(YEAR FROM staging.to_month(r.issue_d))::int       AS issue_year,
        staging.to_num(r.term)::int                               AS term_months,

        -- ===== known at origination =====
        staging.to_num(r.loan_amnt)                               AS loan_amnt,
        staging.to_num(r.funded_amnt)                             AS funded_amnt,
        staging.to_num(r.int_rate)                                AS int_rate,
        staging.to_num(r.installment)                             AS installment,
        r.grade,
        r.sub_grade,
        CASE
            WHEN r.emp_length IS NULL           THEN NULL
            WHEN r.emp_length LIKE '< 1%'       THEN 0
            WHEN r.emp_length LIKE '10+%'       THEN 10
            ELSE staging.to_num(r.emp_length)
        END                                                       AS emp_length_years,
        r.home_ownership,
        staging.to_num(r.annual_inc)                              AS annual_inc,
        r.verification_status,
        r.purpose,
        left(r.zip_code, 3)                                       AS zip3,
        r.addr_state,
        staging.to_num(r.dti)                                     AS dti,
        staging.to_num(r.delinq_2yrs)                             AS delinq_2yrs,
        staging.to_month(r.earliest_cr_line)                      AS earliest_cr_line_d,
        staging.to_num(r.fico_range_low)                          AS fico_range_low,
        staging.to_num(r.fico_range_high)                         AS fico_range_high,
        staging.to_num(r.inq_last_6mths)                          AS inq_last_6mths,
        staging.to_num(r.mths_since_last_delinq)                  AS mths_since_last_delinq,
        staging.to_num(r.mths_since_last_record)                  AS mths_since_last_record,
        staging.to_num(r.open_acc)                                AS open_acc,
        staging.to_num(r.pub_rec)                                 AS pub_rec,
        staging.to_num(r.revol_bal)                               AS revol_bal,
        staging.to_num(r.revol_util)                              AS revol_util,
        staging.to_num(r.total_acc)                               AS total_acc,
        r.initial_list_status,
        r.application_type,
        staging.to_num(r.mort_acc)                                AS mort_acc,
        staging.to_num(r.pub_rec_bankruptcies)                    AS pub_rec_bankruptcies,
        staging.to_num(r.collections_12_mths_ex_med)              AS collections_12_mths_ex_med,
        staging.to_num(r.acc_now_delinq)                          AS acc_now_delinq,
        staging.to_num(r.tot_coll_amt)                            AS tot_coll_amt,
        staging.to_num(r.tot_cur_bal)                             AS tot_cur_bal,
        staging.to_num(r.total_rev_hi_lim)                        AS total_rev_hi_lim,
        staging.to_num(r.avg_cur_bal)                             AS avg_cur_bal,
        staging.to_num(r.bc_util)                                 AS bc_util,
        staging.to_num(r.num_bc_sats)                             AS num_bc_sats,

        -- ===== only known AFTER issuance (leakage candidates) =====
        staging.to_num(r.out_prncp)                               AS out_prncp,
        staging.to_num(r.total_pymnt)                             AS total_pymnt,
        staging.to_num(r.total_rec_prncp)                         AS total_rec_prncp,
        staging.to_num(r.total_rec_int)                           AS total_rec_int,
        staging.to_num(r.total_rec_late_fee)                      AS total_rec_late_fee,
        staging.to_num(r.recoveries)                              AS recoveries,
        staging.to_num(r.collection_recovery_fee)                 AS collection_recovery_fee,
        staging.to_month(r.last_pymnt_d)                          AS last_pymnt_d,
        staging.to_num(r.last_pymnt_amnt)                         AS last_pymnt_amnt,
        staging.to_month(r.last_credit_pull_d)                    AS last_credit_pull_d,
        staging.to_num(r.last_fico_range_high)                    AS last_fico_range_high,
        staging.to_num(r.last_fico_range_low)                     AS last_fico_range_low,
        r.debt_settlement_flag,

        -- ===== target =====
        r.loan_status,
        CASE
            WHEN r.loan_status ~ 'Charged Off|Default' THEN 1
            WHEN r.loan_status ~ 'Fully Paid'          THEN 0
        END::smallint                                             AS is_default
    FROM raw.loans r
    WHERE r.id ~ '^[0-9]+$'
)
SELECT *
FROM typed
WHERE issue_d      IS NOT NULL
  AND term_months  IS NOT NULL
  AND is_default   IS NOT NULL;

ALTER TABLE staging.loans_clean ADD PRIMARY KEY (loan_id);
CREATE INDEX idx_loans_clean_issue ON staging.loans_clean (issue_year, term_months);
ANALYZE staging.loans_clean;
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/data/clean.py"] = r'''"""Run sql/02_staging.sql: raw.loans -> staging.loans_clean, then print a data summary.

Usage:
    python -m finrisk.data.clean
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from finrisk.config import get_settings
from finrisk.db import execute_sql, get_engine

pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 100)


def main() -> None:
    settings = get_settings()
    engine = get_engine()

    with engine.connect() as c:
        exists = c.execute(text("SELECT to_regclass('raw.loans')")).scalar()
    if exists is None:
        raise SystemExit("raw.loans not found. Run: python -m finrisk.data.ingest --sample")

    print("Building staging.loans_clean ...")
    execute_sql((settings.sql_dir / "02_staging.sql").read_text(encoding="utf-8"))

    with engine.connect() as c:
        n_raw = c.execute(text("SELECT COUNT(*) FROM raw.loans")).scalar()
        n_clean = c.execute(text("SELECT COUNT(*) FROM staging.loans_clean")).scalar()
        status = pd.read_sql(
            text("""SELECT loan_status, COUNT(*) AS n
                    FROM raw.loans GROUP BY 1 ORDER BY 2 DESC"""), c)
        by_year = pd.read_sql(
            text("""SELECT issue_year, term_months, COUNT(*) AS n_loans,
                           ROUND(AVG(is_default)::numeric, 4) AS default_rate
                    FROM staging.loans_clean
                    WHERE issue_year BETWEEN 2010 AND 2018
                    GROUP BY 1, 2 ORDER BY 1, 2"""), c)

    print(f"\nraw rows: {n_raw:,}  ->  staged (resolved outcomes): {n_clean:,}"
          f"  ({n_raw - n_clean:,} excluded)\n")
    print("Loan status in raw data (only Fully Paid / Charged Off / Default are kept):")
    print(status.to_string(index=False))
    print("\nDefault rate by issue year and term:")
    print(by_year.to_string(index=False))
    print("\nNext: python -m finrisk.data.splits")


if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/data/splits.py"] = r'''"""Time-based splits: the single source of truth for train / valid / oot / monitor.

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
from finrisk.db import execute_sql, get_engine

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
    if not (ROOT / "src" / "finrisk" / "config.py").exists():
        sys.exit("Run setup_finrisk.py and apply_phase0.py first, from the repo root.")

    print(f"Applying Phase 1 in {ROOT}\n")
    for rel, content in FILES.items():
        write(rel, content)

    print("\nDone. Next:")
    print("  python -m finrisk.data.ingest --sample")
    print("  python -m finrisk.data.clean")
    print("  python -m finrisk.data.splits")


if __name__ == "__main__":
    main()