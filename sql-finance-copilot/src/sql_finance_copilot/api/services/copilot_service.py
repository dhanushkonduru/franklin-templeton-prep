from __future__ import annotations

from dataclasses import dataclass

from sql_finance_copilot.core.models import QueryArtifact, RetrievedSchemaContext
from sql_finance_copilot.core.orchestrator import SqlCopilot


@dataclass(slots=True)
class CopilotService:
    copilot: SqlCopilot

    def run_query(self, question: str, chart: bool = True) -> QueryArtifact:
        return self.copilot.answer(question, chart=chart)

    def repair_sql(self, question: str, sql: str, error: str) -> str:
        return self.copilot.repair_sql(question=question, sql=sql, error=error)

    def retrieve_schema(self, question: str, top_k: int | None = None) -> RetrievedSchemaContext:
        return self.copilot.retrieve_schema(question=question, top_k=top_k)

    def health(self) -> dict[str, str]:
        return self.copilot.health_status()
