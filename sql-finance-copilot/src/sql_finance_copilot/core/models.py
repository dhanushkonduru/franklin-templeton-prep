from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any


@dataclass(slots=True)
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool = True
    default: str | None = None
    comment: str | None = None


@dataclass(slots=True)
class TableInfo:
    schema_name: str
    table_name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    comment: str | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.schema_name}.{self.table_name}"

    def to_document(self, max_columns: int = 40) -> str:
        cols = self.columns[:max_columns]
        column_text = ", ".join(
            f"{column.name} {column.data_type}{' not null' if not column.nullable else ''}"
            for column in cols
        )
        comment = f" comment: {self.comment}" if self.comment else ""
        return f"table {self.qualified_name}{comment}; columns: {column_text}"


@dataclass(slots=True)
class SchemaDocument:
    doc_id: str
    schema_name: str
    table_name: str
    text: str


@dataclass(slots=True)
class RetrievedSchemaContext:
    question: str
    documents: list[SchemaDocument]

    @property
    def table_names(self) -> list[str]:
        return [document.table_name for document in self.documents]

    @property
    def qualified_table_names(self) -> list[str]:
        return [f"{document.schema_name}.{document.table_name}" for document in self.documents]

    @property
    def prompt_text(self) -> str:
        return "\n".join(document.text for document in self.documents)


@dataclass(slots=True)
class QueryResult:
    rows: list[dict[str, Any]]
    columns: list[str]
    row_count: int
    truncated: bool = False


@dataclass(slots=True)
class ValidationResult:
    sql: str
    referenced_tables: list[str]


@dataclass(slots=True)
class ChartSuggestion:
    chart_type: str | None
    x: str | None = None
    y: str | None = None
    title: str | None = None


@dataclass(slots=True)
class QueryArtifact:
    question: str
    sql: str
    validated_sql: str
    rows: list[dict[str, Any]]
    columns: list[str]
    row_count: int
    truncated: bool
    schema_context: RetrievedSchemaContext
    chart: ChartSuggestion | None = field(default_factory=lambda: ChartSuggestion(chart_type=None))
    error: str | None = None
    repaired: bool = False
    repair_attempts: int = 0
    repair_log: list[dict[str, Any]] = field(default_factory=list)
    executed_at: datetime | None = None


def json_safe_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value
