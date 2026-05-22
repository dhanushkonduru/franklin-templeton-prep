from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from sqlalchemy.exc import TimeoutError

from sql_finance_copilot.db.executor import QueryExecutor


class FakeMappingsResult:
    def __init__(self, rows, keys):
        self._rows = rows
        self._keys = keys

    def fetchall(self):
        return self._rows


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return FakeMappingsResult(self._rows, self.keys())

    def keys(self):
        if not self._rows:
            return []
        return list(self._rows[0].keys())


class FakeConnection:
    def __init__(self, rows=None, execute_error=None):
        self.rows = rows or []
        self.execute_error = execute_error
        self.statements = []

    def begin(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, statement, params=None):
        text_value = getattr(statement, "text", None) or str(statement)
        self.statements.append((text_value, params))
        if self.execute_error is not None:
            raise self.execute_error
        return FakeResult(self.rows)


class FakeEngine:
    def __init__(self, rows=None, execute_error=None):
        self.connection = FakeConnection(rows=rows, execute_error=execute_error)

    def connect(self):
        return self.connection


def test_execute_structured_success_is_json_friendly(monkeypatch):
    engine = FakeEngine(
        rows=[
            {"amount": Decimal("12.34"), "created_at": datetime(2026, 5, 21, 12, 0, 0)},
            {"amount": Decimal("56.78"), "created_at": datetime(2026, 5, 21, 13, 0, 0)},
        ]
    )
    executor = QueryExecutor(engine, max_rows=1, statement_timeout_ms=2500)

    times = iter([100.0, 100.05])
    monkeypatch.setattr("sql_finance_copilot.db.executor.perf_counter", lambda: next(times))

    outcome = executor.execute_structured("SELECT amount, created_at FROM trades")

    assert outcome.error is None
    assert outcome.truncated is True
    assert outcome.row_count == 1
    assert outcome.elapsed_ms == pytest.approx(50.0)
    assert outcome.rows == [{"amount": 12.34, "created_at": "2026-05-21T12:00:00"}]
    payload = outcome.to_dict()
    assert payload["max_rows"] == 1
    assert payload["statement_timeout_ms"] == 2500
    assert payload["error"] is None
    assert "SELECT * FROM (SELECT amount, created_at FROM trades)" in engine.connection.statements[2][0]


def test_execute_structured_rejects_multiple_statements():
    engine = FakeEngine()
    executor = QueryExecutor(engine)

    outcome = executor.execute_structured("SELECT 1; SELECT 2")

    assert outcome.error is not None
    assert outcome.error.kind == "multiple_statements"
    assert outcome.rows == []


def test_execute_structured_wraps_timeout_errors(monkeypatch):
    engine = FakeEngine(execute_error=TimeoutError("statement timeout"))
    executor = QueryExecutor(engine, max_rows=10, statement_timeout_ms=1000)

    times = iter([200.0, 200.2])
    monkeypatch.setattr("sql_finance_copilot.db.executor.perf_counter", lambda: next(times))

    outcome = executor.execute_structured("SELECT * FROM trades")

    assert outcome.error is not None
    assert outcome.error.kind == "timeout"
    assert outcome.error.detail == "statement timeout"
    assert outcome.elapsed_ms == pytest.approx(200.0)
    assert outcome.to_dict()["error"]["kind"] == "timeout"
