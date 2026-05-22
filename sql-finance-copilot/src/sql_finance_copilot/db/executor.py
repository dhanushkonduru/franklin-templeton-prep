from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, TimeoutError

from sql_finance_copilot.core.models import QueryResult, json_safe_value


@dataclass(slots=True)
class QueryExecutionError:
    kind: str
    message: str
    sql: str
    elapsed_ms: float | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "sql": self.sql,
            "elapsed_ms": self.elapsed_ms,
            "detail": self.detail,
        }


@dataclass(slots=True)
class QueryExecutionOutcome:
    sql: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    elapsed_ms: float = 0.0
    max_rows: int = 0
    statement_timeout_ms: int = 0
    error: QueryExecutionError | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sql": self.sql,
            "rows": self.rows,
            "columns": self.columns,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "elapsed_ms": self.elapsed_ms,
            "max_rows": self.max_rows,
            "statement_timeout_ms": self.statement_timeout_ms,
            "error": None if self.error is None else self.error.to_dict(),
        }


class SQLExecutionError(RuntimeError):
    pass


class QueryExecutor:
    def __init__(self, engine: Engine, max_rows: int = 500, statement_timeout_ms: int = 10000):
        self._engine = engine
        self._max_rows = max_rows
        self._statement_timeout_ms = statement_timeout_ms

    def execute(self, sql: str) -> QueryResult:
        outcome = self.execute_structured(sql)
        if outcome.error is not None:
            raise SQLExecutionError(outcome.error.message)
        return QueryResult(
            rows=outcome.rows,
            columns=outcome.columns,
            row_count=outcome.row_count,
            truncated=outcome.truncated,
        )

    def execute_structured(self, sql: str) -> QueryExecutionOutcome:
        started = perf_counter()
        stripped_sql = sql.strip().rstrip(";")
        if not stripped_sql:
            return self._error_outcome("empty_sql", "SQL is empty", stripped_sql, started)

        if ";" in stripped_sql:
            return self._error_outcome(
                "multiple_statements",
                "Multiple statements are not allowed in execution layer",
                stripped_sql,
                started,
            )

        wrapped_sql = f"SELECT * FROM ({stripped_sql}) AS sql_copilot_result LIMIT :max_rows"

        try:
            with self._engine.connect() as connection:
                with connection.begin():
                    # Some dialects (e.g., SQLite) do not support SET TRANSACTION or SET LOCAL.
                    # Only execute these statements for dialects that support them (commonly Postgres).
                    dialect_name = getattr(getattr(self._engine, "dialect", None), "name", "")
                    if dialect_name != "sqlite":
                        connection.execute(text("SET TRANSACTION READ ONLY"))
                        connection.execute(text(f"SET LOCAL statement_timeout = {self._statement_timeout_ms}"))
                    result = connection.execute(text(wrapped_sql), {"max_rows": self._max_rows + 1})
                    rows = result.mappings().fetchall()
                    truncated = len(rows) > self._max_rows
                    rows = rows[: self._max_rows]
                    materialized_rows = [
                        {column: json_safe_value(value) for column, value in dict(row).items()}
                        for row in rows
                    ]
                    elapsed_ms = (perf_counter() - started) * 1000.0
                    return QueryExecutionOutcome(
                        sql=stripped_sql,
                        rows=materialized_rows,
                        columns=list(result.keys()),
                        row_count=len(materialized_rows),
                        truncated=truncated,
                        elapsed_ms=elapsed_ms,
                        max_rows=self._max_rows,
                        statement_timeout_ms=self._statement_timeout_ms,
                    )
        except TimeoutError as exc:
            return self._error_outcome("timeout", "SQL execution timed out", stripped_sql, started, str(exc))
        except SQLAlchemyError as exc:
            return self._error_outcome("database_error", "SQL execution failed", stripped_sql, started, str(exc))

    def execute_dict(self, sql: str) -> dict[str, Any]:
        return self.execute_structured(sql).to_dict()

    def _error_outcome(
        self,
        kind: str,
        message: str,
        stripped_sql: str,
        started: float,
        detail: str | None = None,
    ) -> QueryExecutionOutcome:
        elapsed_ms = (perf_counter() - started) * 1000.0
        return QueryExecutionOutcome(
            sql=stripped_sql,
            elapsed_ms=elapsed_ms,
            max_rows=self._max_rows,
            statement_timeout_ms=self._statement_timeout_ms,
            error=QueryExecutionError(
                kind=kind,
                message=message,
                sql=stripped_sql,
                elapsed_ms=elapsed_ms,
                detail=detail,
            ),
        )
