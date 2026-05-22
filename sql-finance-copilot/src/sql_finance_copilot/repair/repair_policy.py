"""Repair policy for failed SQL statements.

This module implements a small orchestration primitive around the existing
`llm.sql_repair.SQLRepairer` to provide a single place to change repair
policies, backoff, and logging.
"""
from __future__ import annotations

from typing import Optional

from sql_finance_copilot.llm.sql_repair import SQLRepairer


class RepairPolicy:
    def __init__(self, repairer: SQLRepairer, max_attempts: int = 2):
        self.repairer = repairer
        self.max_attempts = max_attempts

    def repair(self, question: str, sql: str, error: str, context, attempt: int) -> str:
        """Return repaired SQL given the context and attempt number.

        This is intentionally tiny; expand with backoff, caching, or
        alternative strategies as needed.
        """
        if attempt >= self.max_attempts:
            raise RuntimeError("Exceeded repair attempts")
        return self.repairer.repair(question, sql, error, context)
