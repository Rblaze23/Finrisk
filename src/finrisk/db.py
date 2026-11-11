"""Database helpers: psycopg2 connection (bulk load / DDL) and SQLAlchemy engine (pandas)."""
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
