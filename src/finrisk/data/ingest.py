"""Load the LendingClub accepted-loans CSV into PostgreSQL (raw.loans).

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
