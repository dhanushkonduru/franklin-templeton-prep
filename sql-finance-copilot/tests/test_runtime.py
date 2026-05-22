from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, text

from sql_finance_copilot.execution.executor import execute_query
from sql_finance_copilot.llm.sql_generator import generate_sql
from sql_finance_copilot.repair.repair_loop import repair_query
from sql_finance_copilot.retrieval.retriever import retrieve_relevant_schema
from sql_finance_copilot.validation.validator import validate_sql


class _FakeChoice:
    def __init__(self, content: str):
        self.message = type("Message", (), {"content": content})()


class _FakeCompletions:
    def __init__(self, content: str):
        self._content = content
        self.last_messages = None

    def create(self, *, model, temperature, messages):
        self.last_messages = messages
        return type("Response", (), {"choices": [_FakeChoice(self._content)]})()


class _FakeChat:
    def __init__(self, content: str):
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content: str):
        self.chat = _FakeChat(content)


def test_retrieval_includes_seeded_sector_value():
    schema = retrieve_relevant_schema("What was the top-performing tech stock in 2023?")

    assert "daily_prices" in schema
    assert "stocks" in schema
    assert "sector: Tech" in schema


def test_validate_sql_allows_read_only_select():
    assert validate_sql("SELECT ticker FROM stocks") is True
    assert validate_sql("DROP TABLE stocks") is False


def test_execute_query_uses_shared_database_engine(monkeypatch, tmp_path):
    database_path = tmp_path / "finance.db"
    engine = create_engine(f"sqlite:///{database_path}")

    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE stocks (ticker TEXT, sector TEXT)"))
        connection.execute(text("INSERT INTO stocks VALUES ('AAPL', 'Tech')"))

    monkeypatch.setattr("sql_finance_copilot.execution.executor.get_engine", lambda: engine)

    result = execute_query("SELECT ticker, sector FROM stocks ORDER BY ticker")

    assert result == {"columns": ["ticker", "sector"], "rows": [["AAPL", "Tech"]]}


def test_generate_sql_uses_question_and_schema(monkeypatch):
    fake_client = _FakeClient("```sql\nSELECT 1\n```")
    monkeypatch.setattr("sql_finance_copilot.llm.sql_generator._client", fake_client)

    sql = generate_sql("show the top stock", "Table: stocks")

    assert sql == "SELECT 1"
    messages = fake_client.chat.completions.last_messages
    assert messages[1]["content"].startswith("QUESTION:\nshow the top stock")
    assert "DATABASE SCHEMA:\nTable: stocks" in messages[1]["content"]


def test_repair_query_strips_code_fences(monkeypatch):
    fake_client = _FakeClient("```sql\nSELECT ticker FROM stocks\n```")
    monkeypatch.setattr("sql_finance_copilot.repair.repair_loop._client", fake_client)

    sql = repair_query("SELECT * FROM stocks", "bad column", "Table: stocks")

    assert sql == "SELECT ticker FROM stocks"