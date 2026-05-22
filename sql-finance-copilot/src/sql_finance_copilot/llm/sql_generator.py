from __future__ import annotations

import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SQL_GENERATION_PROMPT = """
You are an expert financial SQL generation engine.

Generate SQLite SQL only.

STRICT RULES:
- Output ONLY raw SQL
- No markdown
- No explanations
- One query only
- SELECT only

SCHEMA RULES:
- Use ONLY tables listed in DATABASE SCHEMA
- Never invent table names
- Never invent column names

SQLITE RULES:
- Use STRFTIME('%Y', column_name) for year extraction
- Never use EXTRACT()

FINANCIAL REASONING RULES:
- "top-performing stock" means highest percentage return over a time period
- NEVER use (close - open) / open for stock performance ranking
- Use time-series aggregation for performance calculations
- Calculate return using:
    (MAX(close) - MIN(close)) * 100.0 / MIN(close)
- Group by ticker when calculating stock performance
- Rank by return percentage, not raw stock price
- For yearly performance:
    - filter rows using STRFTIME('%Y', date)
    - aggregate prices across the full year

CORRECT EXAMPLE:

WITH yearly_returns AS (
    SELECT
        ticker,
        (
            MAX(close) - MIN(close)
        ) * 100.0 / MIN(close) AS return_pct
    FROM daily_prices
    WHERE STRFTIME('%Y', date) = '2023'
    GROUP BY ticker
)

SELECT *
FROM yearly_returns
ORDER BY return_pct DESC
LIMIT 1;
"""


_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def _extract_sql(content: str) -> str:
    cleaned = content.strip()
    cleaned = cleaned.replace("```sql", "")
    cleaned = cleaned.replace("```", "")
    return cleaned.strip()


def generate_sql(question: str, schema: str) -> str:
    messages = [
        {"role": "system", "content": SQL_GENERATION_PROMPT},
        {
            "role": "user",
            "content": f"QUESTION:\n{question}\n\nDATABASE SCHEMA:\n{schema}",
        },
    ]

    response = _client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=messages,
    )

    return _extract_sql(response.choices[0].message.content)


class SQLGenerator:
    """Compatibility wrapper used by `core.orchestrator`.

    The orchestrator expects a `SQLGenerator(client, model)` instance with a
    `generate(question, schema)` method. This adapter accepts a
    `GroqChatClient`-like client (with `complete(...)`) and delegates to the
    underlying LLM.
    """

    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    def generate(self, question: str, schema: str) -> str:
        messages = [
            {"role": "system", "content": SQL_GENERATION_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nDATABASE SCHEMA:\n{schema}",
            },
        ]
        # GroqChatClient.complete returns content string
        content = self._client.complete(model=self._model, messages=messages)
        return _extract_sql(content)