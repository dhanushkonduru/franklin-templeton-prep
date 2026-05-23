from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from app.redis_client import normalized_event_fingerprint
from app.streams.consumer import RedisStreamConsumer
from app.streams.producer import RedisStreamProducer
from app.types import NormalizedNewsEvent, RawNewsEvent


@dataclass
class FakeRedis:
    def __post_init__(self) -> None:
        self.keys: set[str] = set()
        self.streams: dict[str, list[dict[str, str]]] = {}
        self.acks: list[tuple[str, str, str]] = []

    async def set(self, key: str, value: str, *, nx: bool = False, ex: int | None = None):
        if nx and key in self.keys:
            return None
        self.keys.add(key)
        return True

    async def xadd(self, stream: str, payload: dict[str, str], maxlen: int | None = None, approximate: bool = False):
        self.streams.setdefault(stream, []).append(payload)
        return f"{len(self.streams[stream])}-0"

    async def xgroup_create(self, name: str, groupname: str, id: str = "$", mkstream: bool = True):
        self.streams.setdefault(name, [])

    async def xreadgroup(self, groupname: str, consumername: str, streams: dict[str, str], count: int, block: int):
        stream_name = next(iter(streams))
        items = self.streams.get(stream_name, [])[:count]
        self.streams[stream_name] = self.streams.get(stream_name, [])[count:]
        return [(stream_name, [(f"{index + 1}-0", payload) for index, payload in enumerate(items)])]

    async def xack(self, stream: str, group: str, message_id: str):
        self.acks.append((stream, group, message_id))

    async def xautoclaim(self, *args, **kwargs):
        return ("0-0", [], [])


def _normalized_event() -> NormalizedNewsEvent:
    return NormalizedNewsEvent(
        source="newsapi",
        source_event_id="evt-1",
        title="Apple reports strong earnings",
        body="",
        url="https://example.com/apple",
        published_at=datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc),
        tickers=("AAPL",),
        company_name="Apple Inc.",
        content_hash="hash-1",
        raw_payload={},
    )


@pytest.mark.asyncio
async def test_producer_idempotency_blocks_duplicate_publish():
    redis = FakeRedis()
    producer = RedisStreamProducer(redis)

    first = await producer.publish(_normalized_event())
    second = await producer.publish(_normalized_event())

    assert first == "1-0"
    assert second is None
    assert len(redis.streams["news_stream"]) == 1


@pytest.mark.asyncio
async def test_consumer_batches_and_acks_messages():
    redis = FakeRedis()
    redis.streams["news_stream"] = [
        {
            "source": "newsapi",
            "source_event_id": "evt-1",
            "headline": "apple reports strong earnings",
            "ticker": "AAPL",
            "published_at": "2026-05-23T12:00:00+00:00",
            "url": "https://example.com/apple",
            "content_hash": "hash-1",
        }
    ]

    processed: list[RawNewsEvent] = []

    async def process(event: RawNewsEvent) -> None:
        processed.append(event)

    consumer = RedisStreamConsumer(redis, process, batch_size=10, max_retries=2)
    result = await consumer.consume_batch()

    assert result.processed == 1
    assert result.dead_lettered == 0
    assert len(processed) == 1
    assert redis.acks == [("news_stream", "alert-workers", "1-0")]


@pytest.mark.asyncio
async def test_normalized_event_fingerprint_is_stable():
    event = _normalized_event()
    assert normalized_event_fingerprint(event) == normalized_event_fingerprint(event)
