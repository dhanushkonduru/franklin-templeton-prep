from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from app.ingestion import AsyncNewsIngestionLayer
from app.types import RawNewsEvent


class FakeRedis:
    def __init__(self) -> None:
        self.keys: set[str] = set()
        self.stream_entries: list[tuple[str, dict[str, str]]] = []

    async def set(self, key: str, value: str, *, ex: int | None = None, nx: bool = False):
        if nx and key in self.keys:
            return None
        self.keys.add(key)
        return True

    async def xadd(self, stream: str, payload: dict[str, str], maxlen: int | None = None, approximate: bool = False):
        self.stream_entries.append((stream, payload))
        return f"{len(self.stream_entries)}-0"


@pytest.mark.asyncio
async def test_ingestion_normalizes_to_expected_shape():
    redis = FakeRedis()
    layer = AsyncNewsIngestionLayer(redis)
    event = RawNewsEvent(
        source="alpaca",
        source_event_id="abc",
        title=" Apple Reports Strong Earnings ",
        body="",
        url="https://example.com/apple",
        published_at=datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc),
        tickers=("aapl",),
        company_name=None,
        raw_payload={},
    )

    payload = layer.normalize(event)

    assert payload["headline"] == "apple reports strong earnings"
    assert payload["ticker"] == "AAPL"
    assert payload["published_at"] == "2026-05-23T12:00:00+00:00"
    assert payload["source"] == "alpaca"
    assert payload["url"] == "https://example.com/apple"
    assert payload["source_event_id"] == "abc"
    assert payload["body"] == ""
    assert json.loads(payload["tickers"]) == ["AAPL"]


@pytest.mark.asyncio
async def test_ingestion_deduplicates_before_redis():
    redis = FakeRedis()
    layer = AsyncNewsIngestionLayer(redis)
    payload = {
        "source": "alpaca",
        "source_event_id": "evt-dedupe",
        "headline": "apple reports strong earnings",
        "body": "",
        "ticker": "AAPL",
        "tickers": json.dumps(["AAPL"]),
        "published_at": "2026-05-23T12:00:00+00:00",
        "url": "https://example.com/apple",
        "company_name": "",
    }

    first = await layer.publish(payload)
    second = await layer.publish(payload)

    assert first == "1-0"
    assert second is None
    assert len(redis.stream_entries) == 1
