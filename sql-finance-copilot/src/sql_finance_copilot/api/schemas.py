from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    chart: bool = True


class QueryResponse(BaseModel):
    question: str
    sql: str
    validated_sql: str
    rows: list[dict[str, Any]]
    columns: list[str]
    row_count: int
    truncated: bool
    schema_tables: list[str]
    repaired: bool
    repair_attempts: int = 0
    repair_log: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    executed_at: datetime | None = None
    chart_type: str | None = None
    chart_x: str | None = None
    chart_y: str | None = None
    chart_title: str | None = None


class HealthResponse(BaseModel):
    status: str


class RepairRequest(BaseModel):
    question: str = Field(min_length=1)
    sql: str = Field(min_length=1)
    error: str = Field(min_length=1)


class RepairResponse(BaseModel):
    question: str
    original_sql: str
    repaired_sql: str


class SchemaRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)


class SchemaTable(BaseModel):
    doc_id: str
    schema_name: str
    table_name: str
    text: str


class SchemaResponse(BaseModel):
    question: str
    table_names: list[str]
    qualified_table_names: list[str]
    prompt_text: str
    documents: list[SchemaTable]


class HealthCheckResponse(BaseModel):
    status: str
    checks: dict[str, str]
