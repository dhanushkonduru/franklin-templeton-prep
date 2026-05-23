from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.db import create_engine_and_sessionmaker, init_db
from app.repositories import AlertRepository
from app.schemas import PortfolioAliasCreate, PortfolioHoldingCreate, PortfolioMatchRequest, PortfolioSubsidiaryCreate
from app.services.portfolio_matching import PortfolioMatchingService


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def get(self, key: str):
        return self.values.get(key)

    async def set(self, key: str, value: str, ex: int | None = None):
        self.values[key] = value
        return True

    async def delete(self, key: str):
        self.values.pop(key, None)

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_exact_ticker_match_uses_portfolio_symbols():
    service = PortfolioMatchingService(redis=FakeRedis())
    response = await service.match(
        PortfolioMatchRequest(headline="Apple reports strong earnings", ticker="AAPL", event_type="earnings"),
        portfolio_symbols=["AAPL", "TSLA", "MSFT", "NVDA"],
    )

    assert response.portfolio_hit is True
    assert response.matched_symbol == "AAPL"


@pytest.mark.asyncio
async def test_alias_match_maps_google_to_googl(tmp_path):
    engine, session_factory = create_engine_and_sessionmaker("sqlite+aiosqlite:///:memory:")
    await init_db(engine)

    async with session_factory() as session:
        repo = AlertRepository(session)
        await repo.add_holding(
            PortfolioHoldingCreate(
                portfolio_name="Core",
                ticker="GOOGL",
                company_name="Alphabet Inc.",
                active=True,
            )
        )
        await repo.add_alias(PortfolioAliasCreate(portfolio_symbol="GOOGL", alias="Google", active=True))

    service = PortfolioMatchingService(redis=FakeRedis(), session_factory=session_factory)
    response = await service.match(
        PortfolioMatchRequest(headline="Google unveils new AI model", ticker="", event_type="product_launch"),
    )

    assert response.portfolio_hit is True
    assert response.matched_symbol == "GOOGL"


@pytest.mark.asyncio
async def test_subsidiary_match_maps_waymo_to_googl():
    engine, session_factory = create_engine_and_sessionmaker("sqlite+aiosqlite:///:memory:")
    await init_db(engine)

    async with session_factory() as session:
        repo = AlertRepository(session)
        await repo.add_holding(
            PortfolioHoldingCreate(
                portfolio_name="Core",
                ticker="GOOGL",
                company_name="Alphabet Inc.",
                active=True,
            )
        )
        await repo.add_subsidiary(PortfolioSubsidiaryCreate(portfolio_symbol="GOOGL", subsidiary_name="Waymo", active=True))

    service = PortfolioMatchingService(redis=FakeRedis(), session_factory=session_factory)
    response = await service.match(
        PortfolioMatchRequest(headline="Waymo expands autonomous taxi service", ticker="", event_type="other"),
    )

    assert response.portfolio_hit is True
    assert response.matched_symbol == "GOOGL"


@pytest.mark.asyncio
async def test_match_history_persists_and_cache_is_used():
    engine, session_factory = create_engine_and_sessionmaker("sqlite+aiosqlite:///:memory:")
    await init_db(engine)

    async with session_factory() as session:
        repo = AlertRepository(session)
        await repo.add_holding(
            PortfolioHoldingCreate(
                portfolio_name="Core",
                ticker="AAPL",
                company_name="Apple Inc.",
                active=True,
            )
        )

    redis = FakeRedis()
    service = PortfolioMatchingService(redis=redis, session_factory=session_factory, cache_ttl_seconds=3600)
    request = PortfolioMatchRequest(headline="Apple reports strong earnings", ticker="AAPL", event_type="earnings")

    first = await service.match(request)
    second = await service.match(request)

    assert first.portfolio_hit is True
    assert second.portfolio_hit is True
    assert first.matched_symbol == "AAPL"
    assert second.matched_symbol == "AAPL"

    async with session_factory() as session:
        repo = AlertRepository(session)
        history = await repo.list_portfolio_match_history(limit=10)

    assert len(history) == 2
    assert history[0].portfolio_hit is True