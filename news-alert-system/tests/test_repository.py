from __future__ import annotations

from datetime import datetime, timezone
import pytest

from app.db import create_engine_and_sessionmaker, init_db
from app.repositories import AlertRepository
from app.schemas import PortfolioHoldingCreate
from app.types import NormalizedNewsEvent
from app.utils.text import compute_embedding_hash


@pytest.mark.asyncio
async def test_repository_round_trip_with_sqlite():
    engine, session_factory = create_engine_and_sessionmaker("sqlite+aiosqlite:///:memory:")
    await init_db(engine)

    async with session_factory() as session:
        repository = AlertRepository(session)
        holding = await repository.add_holding(
            PortfolioHoldingCreate(
                portfolio_name="Core",
                ticker="AAPL",
                company_name="Apple Inc.",
                active=True,
                webhook_url=None,
                email_address="ops@example.com",
            )
        )

        holdings = await repository.list_holdings(active_only=False)
        events = await repository.list_recent_events(limit=10)

        assert holding.ticker == "AAPL"
        assert len(holdings) == 1
        assert events == []


@pytest.mark.asyncio
async def test_new_tables_and_optimized_columns():
    engine, session_factory = create_engine_and_sessionmaker("sqlite+aiosqlite:///:memory:")
    await init_db(engine)

    async with session_factory() as session:
        repository = AlertRepository(session)
        now = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
        embedding = [0.1, 0.2, 0.3, 0.4]
        emb_hash = compute_embedding_hash(embedding)

        # 1. Test Alerts CRUD
        alert = await repository.create_alert(
            headline="Tesla announces self-driving fleet",
            ticker="TSLA",
            embedding=embedding,
            event_type="product_launch",
            confidence=0.95,
            publication_time=now,
            processing_latency=0.125,
        )

        assert alert.id is not None
        assert alert.headline == "Tesla announces self-driving fleet"
        assert alert.ticker == "TSLA"
        assert alert.embedding_hash == emb_hash
        assert alert.event_type == "product_launch"
        assert alert.confidence == 0.95
        assert alert.publication_time.replace(tzinfo=timezone.utc) == now
        assert alert.processing_latency == 0.125

        alerts = await repository.list_alerts(ticker="TSLA")
        assert len(alerts) == 1
        assert alerts[0].headline == "Tesla announces self-driving fleet"

        alerts_all = await repository.list_alerts()
        assert len(alerts_all) == 1

        # 2. Test Deduplicated Events CRUD
        dedup_event = await repository.create_deduplicated_event(
            headline="Tesla announces self-driving fleet",
            ticker="TSLA",
            embedding=embedding,
            event_type="product_launch",
            confidence=0.95,
            publication_time=now,
            processing_latency=0.125,
            similarity_score=0.98,
            is_duplicate=True,
            duplicate_of="some-original-id",
        )

        assert dedup_event.id is not None
        assert dedup_event.headline == "Tesla announces self-driving fleet"
        assert dedup_event.ticker == "TSLA"
        assert dedup_event.embedding_hash == emb_hash
        assert dedup_event.similarity_score == 0.98
        assert dedup_event.is_duplicate is True
        assert dedup_event.duplicate_of == "some-original-id"

        dedup_events = await repository.list_deduplicated_events(ticker="TSLA")
        assert len(dedup_events) == 1
        assert dedup_events[0].headline == "Tesla announces self-driving fleet"

        # 3. Test Latency Metrics CRUD
        metric = await repository.record_latency_metric(
            event_id="test-event-id",
            stage="classification",
            headline="Tesla announces self-driving fleet",
            ticker="TSLA",
            embedding=embedding,
            event_type="product_launch",
            confidence=0.95,
            publication_time=now,
            processing_latency=0.045,
        )

        assert metric.id is not None
        assert metric.event_id == "test-event-id"
        assert metric.stage == "classification"
        assert metric.processing_latency == 0.045
        assert metric.embedding_hash == emb_hash

        metrics = await repository.list_latency_metrics(stage="classification")
        assert len(metrics) == 1
        assert metrics[0].event_id == "test-event-id"

        # 4. Test upsert_event with optimized columns
        normalized_event = NormalizedNewsEvent(
            source="test_source",
            source_event_id="source-123",
            title="Microsoft acquires AI startup",
            body="Microsoft announced acquisition of a prominent AI startup.",
            url="https://example.com/msft-ai",
            published_at=now,
            tickers=("MSFT",),
            company_name="Microsoft",
            content_hash="test_content_hash",
            raw_payload={},
        )

        persisted = await repository.upsert_event(
            normalized_event,
            event_type="acquisition",
            confidence=0.88,
            similarity_score=0.1,
            duplicate_of=None,
            embedding=embedding,
            processing_latency=0.250,
        )

        assert persisted.id is not None
        assert persisted.headline == "Microsoft acquires AI startup"
        assert persisted.ticker == "MSFT"
        assert persisted.embedding_hash == emb_hash
        assert persisted.confidence == 0.88
        assert persisted.publication_time.replace(tzinfo=timezone.utc) == now
        assert persisted.processing_latency == 0.250
