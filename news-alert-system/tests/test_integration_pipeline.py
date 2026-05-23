from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.config import settings
from app.db import create_engine_and_sessionmaker, init_db
from app.ingestion import AsyncNewsIngestionLayer
from app.pipeline.orchestrator import NewsPipeline
from app.repositories import AlertRepository
from app.schemas import PortfolioHoldingCreate
from app.streams.consumer import RedisStreamConsumer
from app.types import RawNewsEvent


@dataclass
class InMemoryRedis:
    keys: set[str] = field(default_factory=set)
    strings: dict[str, str] = field(default_factory=dict)
    streams: dict[str, list[tuple[str, dict[str, str]]]] = field(default_factory=dict)
    stream_counter: int = 0
    groups: set[tuple[str, str]] = field(default_factory=set)
    acks: list[tuple[str, str, str]] = field(default_factory=list)

    async def set(self, key: str, value: str, *, ex: int | None = None, nx: bool = False):
        if nx and key in self.keys:
            return None
        self.keys.add(key)
        self.strings[key] = value
        return True

    async def get(self, key: str):
        return self.strings.get(key)

    async def ping(self):
        return True

    async def xadd(self, stream: str, payload: dict[str, str], maxlen: int | None = None, approximate: bool = False):
        self.stream_counter += 1
        message_id = f"{self.stream_counter}-0"
        self.streams.setdefault(stream, []).append((message_id, payload))
        return message_id

    async def xgroup_create(self, name: str, groupname: str, id: str = "$", mkstream: bool = True):
        self.streams.setdefault(name, [])
        self.groups.add((name, groupname))

    async def xreadgroup(self, groupname: str, consumername: str, streams: dict[str, str], count: int, block: int):
        stream_name = next(iter(streams))
        pending = self.streams.get(stream_name, [])
        if not pending:
            return []
        batch = pending[:count]
        self.streams[stream_name] = pending[count:]
        return [(stream_name, batch)]

    async def xack(self, stream: str, group: str, message_id: str):
        self.acks.append((stream, group, message_id))

    async def xautoclaim(self, *args, **kwargs):
        return ("0-0", [], [])

    async def zadd(self, name: str, mapping: dict[str, float]):
        return 1

    async def zremrangebyscore(self, name: str, minimum: float, maximum: float):
        return 0

    async def zrevrangebyscore(self, name: str, maximum: float, minimum: float, start: int = 0, num: int = 0):
        return []

    async def mget(self, keys: list[str]):
        return [None for _ in keys]

    def pipeline(self):
        return _FakePipeline(self)

    async def aclose(self):
        return None


class _FakePipeline:
    def __init__(self, redis: InMemoryRedis) -> None:
        self.redis = redis
        self.commands: list[tuple[str, tuple, dict]] = []

    def set(self, key: str, value: str, ex: int | None = None):
        self.commands.append(("set", (key, value), {"ex": ex}))
        return self

    def zadd(self, name: str, mapping: dict[str, float]):
        self.commands.append(("zadd", (name, mapping), {}))
        return self

    def zremrangebyscore(self, name: str, minimum: float, maximum: float):
        self.commands.append(("zremrangebyscore", (name, minimum, maximum), {}))
        return self

    async def execute(self):
        for command, args, kwargs in self.commands:
            if command == "set":
                await self.redis.set(args[0], args[1], ex=kwargs.get("ex"))
        self.commands.clear()
        return [True] * 3


class StaticSource:
    name = "integration"

    def __init__(self, events: list[RawNewsEvent]) -> None:
        self._events = events

    async def fetch_events(self) -> list[RawNewsEvent]:
        return self._events


@pytest.mark.asyncio
async def test_end_to_end_ingest_stream_process_persist():
    engine, session_factory = create_engine_and_sessionmaker("sqlite+aiosqlite:///:memory:")
    await init_db(engine)

    redis = InMemoryRedis()
    event = RawNewsEvent(
        source="integration",
        source_event_id="integration-1",
        title="Apple reports record earnings",
        body="Apple beat revenue estimates.",
        url="https://example.com/integration-apple",
        published_at=datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc),
        tickers=("AAPL",),
        company_name="Apple Inc.",
        raw_payload={"integration": True},
    )

    async with session_factory() as session:
        repository = AlertRepository(session)
        await repository.add_holding(
            PortfolioHoldingCreate(
                portfolio_name="Core",
                ticker="AAPL",
                company_name="Apple Inc.",
                active=True,
            )
        )

    ingestion = AsyncNewsIngestionLayer(
        redis,
        sources=[StaticSource([event])],
        poll_interval_seconds=0.1,
    )
    stats = await ingestion.poll_once()
    assert stats.published == 1
    assert len(redis.streams[settings.redis_stream]) == 1

    pipeline = NewsPipeline(redis=redis, session_factory=session_factory)
    pipeline.notification_service = AsyncMock()
    pipeline.notification_service.dispatch = AsyncMock(return_value=[])
    processed_events: list[RawNewsEvent] = []

    async def capture_process(raw: RawNewsEvent):
        processed_events.append(raw)
        await pipeline.process_raw_event(raw)

    consumer = RedisStreamConsumer(redis, capture_process, batch_size=5, max_retries=2)
    result = await consumer.consume_batch()

    assert result.processed == 1
    assert len(processed_events) == 1
    assert "apple beat revenue estimates" in processed_events[0].body
    assert processed_events[0].tickers == ("AAPL",)

    async with session_factory() as session:
        repository = AlertRepository(session)
        events = await repository.list_recent_events(limit=10)
        alerts = await repository.list_alerts(limit=10)
        dedup_rows = await repository.list_deduplicated_events(limit=10)

    assert len(events) == 1
    assert events[0].event_type in {"earnings", "other", "analyst_rating"}
    assert len(dedup_rows) == 1
    assert dedup_rows[0].is_duplicate is False
    assert len(alerts) >= 1
