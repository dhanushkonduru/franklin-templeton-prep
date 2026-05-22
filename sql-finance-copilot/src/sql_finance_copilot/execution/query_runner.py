"""Query runner adapter that wraps the lower-level executor.

The project already contains a robust `QueryExecutor` in `db.executor`.
This adapter gives a clear place for higher-level execution policies and
instrumentation to live in the `execution` package.
"""
from __future__ import annotations

from typing import Any

from sql_finance_copilot.db.executor import QueryExecutor


class QueryRunner:
    def __init__(self, executor: QueryExecutor):
        self._executor = executor

    def run(self, sql: str) -> Any:
        """Execute SQL using the underlying QueryExecutor and return the result object."""
        return self._executor.execute(sql)
