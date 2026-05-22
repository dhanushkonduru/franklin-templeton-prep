from __future__ import annotations

import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
REPAIR_PROMPT = """
You are a SQLite financial SQL repair engine.

The following SQL query failed.

FAILED QUERY:
{sql}

DATABASE ERROR:
{error}

AVAILABLE DATABASE SCHEMA:
{schema}

TASK:
Fix the SQL query while preserving the original business intent.

STRICT RULES:
- Return ONLY corrected SQL
- No explanations
- No markdown
- One query only
- SELECT only

SCHEMA RULES:
- Use ONLY tables listed in AVAILABLE DATABASE SCHEMA
- Use ONLY columns listed in AVAILABLE DATABASE SCHEMA
- Never invent table names
- Never invent column names

SQLITE RULES:
- Use STRFTIME('%Y', column_name) for year filtering
- Never use EXTRACT()
- Generate valid SQLite syntax only

FINANCIAL REASONING RULES:
- "top-performing stock" means highest percentage return over the full period
- NEVER use:
    (close - open) / open
  for stock performance ranking
- Use yearly aggregated return logic:
    (MAX(close) - MIN(close)) * 100.0 / MIN(close)
- Group by ticker for performance calculations
- Rank by percentage return, not raw stock price

CORRECT PATTERN EXAMPLE:

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


def repair_query(sql: str, error: str, schema: str) -> str:
    prompt = REPAIR_PROMPT.format(sql=sql, error=error, schema=schema)

    response = _client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    repaired_sql = response.choices[0].message.content.strip()
    repaired_sql = repaired_sql.replace("```sql", "")
    repaired_sql = repaired_sql.replace("```", "")

    return repaired_sql.strip()