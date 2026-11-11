"""Central configuration for FinRisk Lab.

All settings come from environment variables, loaded from the .env file at the
project root. Use `get_settings()` everywhere instead of reading os.environ.

Check your setup with:  python -m finrisk.config
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _get(name: str, default: str) -> str:
    return os.getenv(name, default)


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # PostgreSQL
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str
    postgres_port: int
    # MLflow
    mlflow_tracking_uri: str
    # Groq
    groq_api_key: str
    groq_model: str
    # Data
    raw_data_path: Path
    loan_term_months: int
    # Time-based split (loan issue years)
    train_start_year: int
    train_end_year: int
    valid_year: int
    oot_year: int
    monitor_year: int

    def __post_init__(self) -> None:
        if not (
            self.train_start_year
            <= self.train_end_year
            < self.valid_year
            < self.oot_year
            < self.monitor_year
        ):
            raise ValueError(
                "Split years must satisfy "
                "TRAIN_START <= TRAIN_END < VALID < OOT < MONITOR, got "
                f"{self.train_start_year}, {self.train_end_year}, "
                f"{self.valid_year}, {self.oot_year}, {self.monitor_year}"
            )
        if self.loan_term_months not in (36, 60):
            raise ValueError("LOAN_TERM_MONTHS must be 36 or 60")

    # ---- derived values ---------------------------------------------------
    @property
    def database_url(self) -> str:
        """SQLAlchemy URL for the finrisk database (used from the host machine)."""
        return (
            f"postgresql+psycopg2://{quote_plus(self.postgres_user)}:"
            f"{quote_plus(self.postgres_password)}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def sql_dir(self) -> Path:
        return PROJECT_ROOT / "sql"

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / "reports"

    @property
    def has_groq_key(self) -> bool:
        return bool(self.groq_api_key) and self.groq_api_key != "your_key_here"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    raw_path = Path(_get("RAW_DATA_PATH", "data/raw/accepted_2007_to_2018Q4.csv.gz"))
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path

    return Settings(
        postgres_user=_get("POSTGRES_USER", "finrisk"),
        postgres_password=_get("POSTGRES_PASSWORD", "finrisk"),
        postgres_db=_get("POSTGRES_DB", "finrisk"),
        postgres_host=_get("POSTGRES_HOST", "localhost"),
        postgres_port=_get_int("POSTGRES_PORT", 5432),
        mlflow_tracking_uri=_get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        groq_api_key=_get("GROQ_API_KEY", ""),
        groq_model=_get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        raw_data_path=raw_path,
        loan_term_months=_get_int("LOAN_TERM_MONTHS", 36),
        train_start_year=_get_int("TRAIN_START_YEAR", 2012),
        train_end_year=_get_int("TRAIN_END_YEAR", 2013),
        valid_year=_get_int("VALID_YEAR", 2014),
        oot_year=_get_int("OOT_YEAR", 2015),
        monitor_year=_get_int("MONITOR_YEAR", 2016),
    )


if __name__ == "__main__":
    s = get_settings()
    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Database         : {s.postgres_user}@{s.postgres_host}:{s.postgres_port}/{s.postgres_db}")
    print(f"MLflow           : {s.mlflow_tracking_uri}")
    print(f"Groq key set     : {s.has_groq_key}  (model: {s.groq_model})")
    print(f"Raw data         : {s.raw_data_path}  (exists: {s.raw_data_path.exists()})")
    print(f"Loan term        : {s.loan_term_months} months")
    print(
        f"Splits           : train {s.train_start_year}-{s.train_end_year} | "
        f"valid {s.valid_year} | oot {s.oot_year} | monitor {s.monitor_year}"
    )
