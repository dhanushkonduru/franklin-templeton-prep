from __future__ import annotations

from sqlalchemy import text

from sql_finance_copilot.database.engine import get_engine


def execute_query(query: str):
    engine = get_engine()

    with engine.connect() as connection:
        result = connection.execute(text(query))
        rows = result.fetchall()
        columns = result.keys()

    return {
        "columns": list(columns),
        "rows": [list(row) for row in rows],
    }