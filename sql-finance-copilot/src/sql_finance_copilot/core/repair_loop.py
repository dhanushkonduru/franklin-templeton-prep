from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from sql_finance_copilot.core.models import QueryResult, RetrievedSchemaContext, ValidationResult
from sql_finance_copilot.db.executor import QueryExecutionOutcome, QueryExecutor
from sql_finance_copilot.llm.sql_repair import SQLRepairer
from sql_finance_copilot.validation.sql_safety import SQLSafetyError, SQLSafetyValidator

LOG = logging.getLogger("sql_repair_loop")


@dataclass(slots=True)
class RepairAttempt:
    attempt: int
    sql: str
    error: str
    repaired_sql: str | None = None


@dataclass(slots=True)
class RepairLoopResult:
    question: str
    original_sql: str
    validated_sql: str | None = None
    query_result: QueryResult | None = None
    attempts: int = 0
    repaired: bool = False
    error: str | None = None
    attempt_log: list[RepairAttempt] = field(default_factory=list)


class RepairLoopError(RuntimeError):
    pass


class SQLRepairRetryLoop:
    def __init__(
        self,
        validator: SQLSafetyValidator,
        executor: QueryExecutor,
        repairer: SQLRepairer,
        max_attempts: int = 2,
        max_rows: int | None = None,
        logger: logging.Logger | None = None,
    ):
        self._validator = validator
        self._executor = executor
        self._repairer = repairer
        self._max_attempts = max(0, max_attempts)
        self._max_rows = max_rows if max_rows is not None else getattr(executor, "_max_rows", 100)
        self._logger = logger or LOG

    def run(
        self,
        question: str,
        initial_sql: str,
        context: RetrievedSchemaContext,
        allowed_tables: Sequence[str],
        allowed_columns: dict[str, Sequence[str]] | None = None,
    ) -> RepairLoopResult:
        current_sql = self._normalize_sql(initial_sql)
        seen_sql = {current_sql}
        attempt_log: list[RepairAttempt] = []
        first_error: str | None = None

        for attempt in range(self._max_attempts + 1):
            self._logger.info(
                "sql repair attempt=%s question=%r sql=%r",
                attempt,
                question,
                current_sql,
            )
            try:
                validated = self._validate(current_sql, allowed_tables, allowed_columns)
                outcome = self._executor.execute_structured(validated.sql)
                if outcome.error is not None:
                    raise RepairLoopError(self._format_execution_error(outcome))

                self._logger.info(
                    "sql repair success attempts=%s rows=%s truncated=%s",
                    attempt,
                    outcome.row_count,
                    outcome.truncated,
                )
                return RepairLoopResult(
                    question=question,
                    original_sql=initial_sql,
                    validated_sql=validated.sql,
                    query_result=QueryResult(
                        rows=outcome.rows,
                        columns=outcome.columns,
                        row_count=outcome.row_count,
                        truncated=outcome.truncated,
                    ),
                    attempts=attempt,
                    repaired=attempt > 0,
                    error=first_error,
                    attempt_log=attempt_log,
                )
            except (SQLSafetyError, RepairLoopError) as exc:
                current_error = str(exc)
                if first_error is None:
                    first_error = current_error

                if attempt >= self._max_attempts:
                    self._logger.warning(
                        "sql repair exhausted attempts=%s error=%s",
                        attempt,
                        current_error,
                    )
                    return RepairLoopResult(
                        question=question,
                        original_sql=initial_sql,
                        validated_sql=None,
                        query_result=None,
                        attempts=attempt,
                        repaired=attempt > 0,
                        error=current_error,
                        attempt_log=attempt_log,
                    )

                try:
                    repaired_sql = self._repair_sql(question, current_sql, current_error, context)
                except Exception as exc:  # noqa: BLE001 - convert repair failures into a safe terminal result
                    self._logger.exception("sql repair model failed attempt=%s", attempt)
                    failure_error = f"Repair model failure: {exc}"
                    return RepairLoopResult(
                        question=question,
                        original_sql=initial_sql,
                        validated_sql=None,
                        query_result=None,
                        attempts=attempt,
                        repaired=attempt > 0,
                        error=failure_error,
                        attempt_log=attempt_log,
                    )

                attempt_log.append(
                    RepairAttempt(attempt=attempt, sql=current_sql, error=current_error, repaired_sql=repaired_sql)
                )
                self._logger.info(
                    "sql repair produced attempt=%s changed=%s sql=%r",
                    attempt,
                    repaired_sql != current_sql,
                    repaired_sql,
                )
                if not repaired_sql:
                    self._logger.warning("sql repair returned empty sql attempt=%s", attempt)
                    return RepairLoopResult(
                        question=question,
                        original_sql=initial_sql,
                        validated_sql=None,
                        query_result=None,
                        attempts=attempt,
                        repaired=attempt > 0,
                        error="Repair model returned empty SQL",
                        attempt_log=attempt_log,
                    )
                repaired_sql = self._normalize_sql(repaired_sql)
                if repaired_sql == current_sql:
                    self._logger.warning("sql repair returned unchanged sql attempt=%s", attempt)
                    return RepairLoopResult(
                        question=question,
                        original_sql=initial_sql,
                        validated_sql=None,
                        query_result=None,
                        attempts=attempt,
                        repaired=attempt > 0,
                        error="Repair model returned unchanged SQL",
                        attempt_log=attempt_log,
                    )
                if repaired_sql in seen_sql:
                    self._logger.warning("sql repair repeated prior sql attempt=%s", attempt)
                    return RepairLoopResult(
                        question=question,
                        original_sql=initial_sql,
                        validated_sql=None,
                        query_result=None,
                        attempts=attempt,
                        repaired=attempt > 0,
                        error="Repair loop detected repeated SQL",
                        attempt_log=attempt_log,
                    )
                seen_sql.add(repaired_sql)
                current_sql = repaired_sql

        raise RepairLoopError("Repair loop terminated unexpectedly")

    def _validate(
        self,
        sql: str,
        allowed_tables: Sequence[str],
        allowed_columns: dict[str, Sequence[str]] | None,
    ) -> ValidationResult:
        return self._validator.validate(
            sql,
            list(allowed_tables),
            allowed_columns=allowed_columns,
            max_rows=self._max_rows,
        )

    def _repair_sql(self, question: str, current_sql: str, current_error: str, context: RetrievedSchemaContext) -> str:
        return self._repairer.repair(question, current_sql, current_error, context)

    def _format_execution_error(self, outcome: QueryExecutionOutcome) -> str:
        if outcome.error is None:
            return ""
        detail = f"; detail={outcome.error.detail}" if outcome.error.detail else ""
        elapsed_ms = outcome.error.elapsed_ms if outcome.error.elapsed_ms is not None else 0.0
        return (
            f"{outcome.error.message}"
            f" (kind={outcome.error.kind}, elapsed_ms={elapsed_ms:.2f}{detail})"
        )

    def _normalize_sql(self, sql: str) -> str:
        return sql.strip().rstrip(";")


__all__ = ["RepairAttempt", "RepairLoopError", "RepairLoopResult", "SQLRepairRetryLoop"]
