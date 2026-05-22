from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


load_dotenv()


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite:///finance.db")


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(get_database_url(), future=True)
