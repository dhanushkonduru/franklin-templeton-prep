from __future__ import annotations

import os
from dataclasses import dataclass, field

import pytest
from sqlalchemy import create_engine, text

from sql_finance_copilot.core.models import RetrievedSchemaContext, SchemaDocument


@dataclass
class FakeGroqClient:
    responses: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, *, model: str, messages: list[dict[str, str]], temperature: float = 0.0, max_tokens: int = 800) -> str:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self.responses:
            raise RuntimeError("No mock Groq response available")
        return self.responses.pop(0)


@pytest.fixture
def fake_groq_client() -> FakeGroqClient:
    return FakeGroqClient()


@pytest.fixture
def schema_context() -> RetrievedSchemaContext:
    return RetrievedSchemaContext(
        question="show revenue",
        documents=[
            SchemaDocument(
                doc_id="public.financials",
                schema_name="public",
                table_name="financials",
                text="table public.financials; columns: ticker text, revenue numeric, net_income numeric, eps numeric",
            )
        ],
    )


@pytest.fixture(scope="session")
def postgres_test_url() -> str | None:
    return os.getenv("TEST_DATABASE_URL")


@pytest.fixture
def postgres_engine(postgres_test_url: str | None):
    if not postgres_test_url:
        pytest.skip("Set TEST_DATABASE_URL to run Postgres integration tests")

    engine = create_engine(postgres_test_url, future=True)
    with engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS public"))
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS public.stocks (
                  stock_id SERIAL PRIMARY KEY,
                  ticker TEXT UNIQUE NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS public.financials (
                  financial_id SERIAL PRIMARY KEY,
                  stock_id INT REFERENCES public.stocks(stock_id),
                  period_end DATE NOT NULL,
                  statement_type TEXT NOT NULL,
                  revenue NUMERIC,
                  net_income NUMERIC,
                  shares_outstanding BIGINT
                )
                """
            )
        )
        connection.execute(text("TRUNCATE TABLE public.financials, public.stocks RESTART IDENTITY CASCADE"))
        connection.execute(text("INSERT INTO public.stocks (ticker) VALUES ('AAPL'), ('MSFT')"))

    try:
        yield engine
    finally:
        engine.dispose()
