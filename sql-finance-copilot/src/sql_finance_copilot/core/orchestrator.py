from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.engine import Engine

from sql_finance_copilot.charts.builder import ChartBuilder
from sql_finance_copilot.config import AppSettings
from sql_finance_copilot.core.models import QueryArtifact
from sql_finance_copilot.core.repair_loop import SQLRepairRetryLoop
from sql_finance_copilot.db.engine import create_db_engine
from sql_finance_copilot.db.executor import QueryExecutor
from sql_finance_copilot.llm.groq_client import GroqChatClient
from sql_finance_copilot.llm.sql_generator import SQLGenerator
from sql_finance_copilot.llm.sql_repair import SQLRepairer
from sql_finance_copilot.schema.retriever import SchemaRetriever
from sql_finance_copilot.validation.sql_safety import SQLSafetyValidator


@dataclass(slots=True)
class OrchestratorDependencies:
    settings: AppSettings
    engine: Engine
    retriever: SchemaRetriever
    generator: SQLGenerator
    repairer: SQLRepairer
    validator: SQLSafetyValidator
    executor: QueryExecutor
    chart_builder: ChartBuilder
    repair_loop: SQLRepairRetryLoop | None = None


class SqlCopilot:
    def __init__(self, dependencies: OrchestratorDependencies):
        self._dependencies = dependencies

    @classmethod
    def build(cls, settings: AppSettings) -> "SqlCopilot":
        engine = create_db_engine(settings)
        retriever = SchemaRetriever(engine, settings)
        retriever.initialize()
        client = GroqChatClient(settings)
        dependencies = OrchestratorDependencies(
            settings=settings,
            engine=engine,
            retriever=retriever,
            generator=SQLGenerator(client, settings.groq_model),
            repairer=SQLRepairer(client, settings.groq_repair_model),
            validator=SQLSafetyValidator(),
            executor=QueryExecutor(engine, settings.max_result_rows, settings.statement_timeout_ms),
            chart_builder=ChartBuilder(),
        )
        dependencies.repair_loop = SQLRepairRetryLoop(
            validator=dependencies.validator,
            executor=dependencies.executor,
            repairer=dependencies.repairer,
            max_attempts=settings.max_repair_attempts,
            max_rows=settings.max_result_rows,
        )
        return cls(dependencies)

    def answer(self, question: str, chart: bool = True) -> QueryArtifact:
        schema_context = self._dependencies.retriever.retrieve(question)
        generated_sql = self._dependencies.generator.generate(question, schema_context)
        allowed_tables = schema_context.qualified_table_names + schema_context.table_names
        if self._dependencies.repair_loop is None:
            raise RuntimeError("Repair loop is not configured")
        repair_result = self._dependencies.repair_loop.run(
            question=question,
            initial_sql=generated_sql,
            context=schema_context,
            allowed_tables=allowed_tables,
        )

        if repair_result.query_result is None:
            raise RuntimeError(repair_result.error or "Query execution failed after repair attempts")

        chart_result = self._dependencies.chart_builder.suggest(repair_result.query_result.rows, title=question) if chart else None
        return QueryArtifact(
            question=question,
            sql=generated_sql,
            validated_sql=repair_result.validated_sql or generated_sql,
            rows=repair_result.query_result.rows,
            columns=repair_result.query_result.columns,
            row_count=repair_result.query_result.row_count,
            truncated=repair_result.query_result.truncated,
            schema_context=schema_context,
            chart=chart_result.suggestion if chart_result else None,
            error=repair_result.error,
            repaired=repair_result.repaired,
            repair_attempts=repair_result.attempts,
            repair_log=[
                {
                    "attempt": attempt.attempt,
                    "sql": attempt.sql,
                    "error": attempt.error,
                    "repaired_sql": attempt.repaired_sql,
                }
                for attempt in repair_result.attempt_log
            ],
            executed_at=datetime.now(timezone.utc),
        )

    def retrieve_schema(self, question: str, top_k: int | None = None):
        return self._dependencies.retriever.retrieve(question, top_k=top_k)

    def repair_sql(self, question: str, sql: str, error: str) -> str:
        context = self._dependencies.retriever.retrieve(question)
        return self._dependencies.repairer.repair(question=question, original_sql=sql, error_message=error, context=context)

    def health_status(self) -> dict[str, str]:
        checks = {
            "app": "ok",
            "database": "unknown",
            "schema_index": "ok" if self._dependencies.retriever is not None else "unknown",
            "llm": "configured" if bool(self._dependencies.settings.groq_api_key) else "missing_api_key",
        }
        try:
            with self._dependencies.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            checks["database"] = "ok"
        except Exception:
            checks["database"] = "error"
        return checks
