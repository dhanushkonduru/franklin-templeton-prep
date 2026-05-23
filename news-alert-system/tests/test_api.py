from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import create_app
from app.db import create_engine_and_sessionmaker, init_db


@pytest.fixture()
async def api_client():
    engine, session_factory = create_engine_and_sessionmaker("sqlite+aiosqlite:///:memory:")
    await init_db(engine)

    app = create_app()

    async def override_get_session():
        async with session_factory() as session:
            yield session

    redis = AsyncMock()
    redis.ping = AsyncMock(return_value=True)

    async def override_get_redis():
        return redis

    from app.db import get_session
    from app.redis_client import get_redis_client

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_redis_client] = override_get_redis

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, redis

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health_live_endpoint(api_client):
    client, _redis = api_client
    response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_health_ready_endpoint(api_client):
    client, _redis = api_client
    response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_metrics_endpoint(api_client):
    client, _redis = api_client
    response = await client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "pipeline_metrics" in body
    assert body["pipeline_metrics"]["total_events_processed"] == 0


@pytest.mark.asyncio
async def test_ingest_endpoint(api_client):
    client, redis = api_client
    redis.set = AsyncMock(return_value=True)

    layer_publish = AsyncMock(return_value="1-0")
    from app import api as api_module

    original_layer = api_module.AsyncNewsIngestionLayer

    class StubLayer:
        def __init__(self, _redis):
            pass

        async def publish_normalized_event(self, _event):
            return await layer_publish()

    api_module.AsyncNewsIngestionLayer = StubLayer

    response = await client.post(
        "/events/ingest",
        json={
            "source": "manual",
            "source_event_id": "manual-1",
            "title": "Tesla launches new product",
            "body": "Product details announced.",
            "url": "https://example.com/tesla",
            "published_at": datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc).isoformat(),
            "tickers": ["TSLA"],
        },
    )

    api_module.AsyncNewsIngestionLayer = original_layer

    assert response.status_code == 202
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["stream_id"] == "1-0"
