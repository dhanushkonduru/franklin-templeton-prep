from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from sql_finance_copilot.config import AppSettings


def create_db_engine(settings: AppSettings) -> Engine:
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        future=True,
    )
