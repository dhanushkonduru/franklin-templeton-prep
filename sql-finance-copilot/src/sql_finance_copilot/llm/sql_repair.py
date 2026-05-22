from __future__ import annotations

import re

from sql_finance_copilot.core.models import RetrievedSchemaContext
from sql_finance_copilot.llm.groq_client import GroqChatClient
from sql_finance_copilot.llm.prompts import SQL_REPAIR_PROMPT


def _extract_sql(text: str) -> str:
    cleaned = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    cleaned = cleaned.removeprefix("sql").strip()
    return cleaned.rstrip(";")


class SQLRepairer:
    def __init__(self, client: GroqChatClient, model: str):
        self._client = client
        self._model = model

    def repair(self, question: str, original_sql: str, error_message: str, context: RetrievedSchemaContext) -> str:
        messages = [
            {"role": "system", "content": SQL_REPAIR_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Original SQL:\n{original_sql}\n\n"
                    f"Execution or validation error:\n{error_message}\n\n"
                    f"Relevant schema:\n{context.prompt_text}\n\n"
                    "Return the corrected SQL only."
                ),
            },
        ]
        content = self._client.complete(model=self._model, messages=messages)
        return _extract_sql(content)
