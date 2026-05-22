from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv


# Load .env into the environment early so `AppSettings.from_env()` picks values
load_dotenv()


def _parse_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(slots=True)
class AppSettings:
    # Default to a local sqlite file for easy local demos. Override with DATABASE_URL for Postgres.
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///finance.db"))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    groq_repair_model: str = field(default_factory=lambda: os.getenv("GROQ_REPAIR_MODEL", "llama-3.3-70b-versatile"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    faiss_index_path: Path = field(default_factory=lambda: Path(os.getenv("FAISS_INDEX_PATH", ".data/schema_index")))
    allowed_schemas: list[str] = field(default_factory=lambda: _parse_csv(os.getenv("ALLOWED_SCHEMAS"), ["public"]))
    max_schema_tables: int = field(default_factory=lambda: int(os.getenv("MAX_SCHEMA_TABLES", "8")))
    max_schema_columns_per_table: int = field(default_factory=lambda: int(os.getenv("MAX_SCHEMA_COLUMNS_PER_TABLE", "40")))
    max_repair_attempts: int = field(default_factory=lambda: int(os.getenv("MAX_REPAIR_ATTEMPTS", "2")))
    max_result_rows: int = field(default_factory=lambda: int(os.getenv("MAX_RESULT_ROWS", "500")))
    statement_timeout_ms: int = field(default_factory=lambda: int(os.getenv("STATEMENT_TIMEOUT_MS", "10000")))
    max_request_body_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_REQUEST_BODY_BYTES", "1048576")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls()

    def ensure_paths(self) -> None:
        self.faiss_index_path.mkdir(parents=True, exist_ok=True)
