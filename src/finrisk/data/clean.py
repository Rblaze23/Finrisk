"""Run sql/02_staging.sql: raw.loans -> staging.loans_clean, then print a data summary.

Usage:
    python -m finrisk.data.clean
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from finrisk.config import get_settings
from finrisk.db import connect, execute_sql, get_engine

pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 100)


def main() -> None:
    settings = get_settings()
    connect().close()  # friendly error if Docker is down
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
