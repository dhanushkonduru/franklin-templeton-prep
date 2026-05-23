from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Awaitable, Callable

from redis.asyncio import Redis

from app.config import settings
from app.redis_client import ensure_stream_groups
from app.types import RawNewsEvent
from app.utils.retry import async_retry


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ConsumerResult:
    processed: int = 0
    retried: int = 0
    dead_lettered: int = 0


class RedisStreamConsumer:
    def __init__(
        self,
        redis: Redis,
        process: Callable[[RawNewsEvent], Awaitable[None]],
        *,
        stream_name: str | None = None,
        dlq_stream_name: str | None = None,
        consumer_group: str | None = None,
        consumer_name: str | None = None,
        batch_size: int | None = None,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
    ) -> None:
        self.redis = redis
        self.process = process
        self.stream_name = stream_name or settings.redis_stream
        self.dlq_stream_name = dlq_stream_name or settings.redis_dlq_stream
        self.consumer_group = consumer_group or settings.redis_consumer_group
        self.consumer_name = consumer_name or settings.redis_consumer_name
        self.batch_size = batch_size or settings.redis_consumer_batch_size
        self.max_retries = max_retries or settings.redis_consumer_max_retries
        self.retry_backoff_seconds = retry_backoff_seconds or settings.redis_consumer_retry_backoff_seconds

    async def ensure_groups(self) -> None:
        await ensure_stream_groups(self.redis, stream=self.stream_name, dlq_stream=self.dlq_stream_name, group=self.consumer_group)

    def _decode_tickers(self, payload: dict[str, str]) -> tuple[str, ...]:
        ticker = payload.get("ticker", "").strip()
        tickers_raw = payload.get("tickers", "")
        parsed: list[str] = []
        if tickers_raw:
            try:
                loaded = json.loads(tickers_raw)
                if isinstance(loaded, list):
                    parsed = [str(item).strip() for item in loaded if str(item).strip()]
            except json.JSONDecodeError:
                parsed = [tickers_raw.strip()] if tickers_raw.strip() else []
        if ticker and ticker not in parsed:
            parsed.insert(0, ticker)
        return tuple(parsed)

    def _decode_event(self, payload: dict[str, str]) -> RawNewsEvent:
        published_at = datetime.fromisoformat(payload["published_at"])
        headline = payload.get("headline", "")
        source = payload.get("source", "unknown")
        url = payload.get("url", "")
        source_event_id = payload.get("source_event_id") or sha256(
            f"{source}|{headline}|{published_at.isoformat()}|{url}".encode("utf-8")
        ).hexdigest()
        company_name = payload.get("company_name") or None
        if company_name == "":
            company_name = None
        raw_payload = {key: value for key, value in payload.items()}
        return RawNewsEvent(
            source=source,
            source_event_id=source_event_id,
            title=headline,
            body=payload.get("body", ""),
            url=url,
            published_at=published_at,
            tickers=self._decode_tickers(payload),
            company_name=company_name,
            raw_payload=raw_payload,
        )

    async def _reclaim_stale_messages(self) -> list[tuple[str, dict[str, str]]]:
        try:
            response = await self.redis.xautoclaim(
                name=self.stream_name,
                groupname=self.consumer_group,
                consumername=self.consumer_name,
                min_idle_time=60_000,
                start_id="0-0",
                count=self.batch_size,
            )
        except Exception:
            return []

        if isinstance(response, tuple) and len(response) >= 2:
            return [(message_id, payload) for message_id, payload in response[1]]
        return []

    async def _move_to_dlq(self, message_id: str, payload: dict[str, str], error: Exception, attempts: int) -> None:
        dlq_payload = {
            **payload,
            "failed_message_id": message_id,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "attempts": str(attempts),
            "error": str(error),
        }
        await self.redis.xadd(self.dlq_stream_name, dlq_payload, maxlen=10_000, approximate=True)
        await self.redis.xack(self.stream_name, self.consumer_group, message_id)
        logger.error(
            "message moved to dead letter queue",
            extra={"stream": self.stream_name, "dlq_stream": self.dlq_stream_name, "message_id": message_id, "attempts": attempts, "error": str(error)},
        )

    async def _process_message(self, message_id: str, payload: dict[str, str]) -> tuple[bool, bool]:
        attempts = int(payload.get("attempts", "0"))
        event = self._decode_event(payload)

        async def _attempt() -> None:
            await self.process(event)

        try:
            await async_retry(
                _attempt,
                attempts=min(self.max_retries, max(1, self.max_retries - attempts)),
                base_delay=self.retry_backoff_seconds,
                operation_name="redis_stream_process",
            )
            await self.redis.xack(self.stream_name, self.consumer_group, message_id)
            logger.info(
                "message processed",
                extra={"stream": self.stream_name, "message_id": message_id, "source": event.source, "source_event_id": event.source_event_id},
            )
            return True, False
        except Exception as exc:
            attempts += 1
            if attempts >= self.max_retries:
                await self._move_to_dlq(message_id, payload, exc, attempts)
                return False, True

            await self.redis.xadd(self.stream_name, {**payload, "attempts": str(attempts)}, maxlen=10_000, approximate=True)
            await self.redis.xack(self.stream_name, self.consumer_group, message_id)
            logger.warning(
                "message scheduled for retry",
                extra={"stream": self.stream_name, "message_id": message_id, "attempts": attempts, "error": str(exc)},
            )
            return False, False

    async def consume_batch(self) -> ConsumerResult:
        await self.ensure_groups()
        result = ConsumerResult()
        messages = await self.redis.xreadgroup(
            groupname=self.consumer_group,
            consumername=self.consumer_name,
            streams={self.stream_name: ">"},
            count=self.batch_size,
            block=1000,
        )
        batch: list[tuple[str, dict[str, str]]] = []
        for _, stream_messages in messages:
            batch.extend(stream_messages)
        if not batch:
            batch = await self._reclaim_stale_messages()

        if not batch:
            return result

        concurrency = min(self.batch_size, settings.pipeline_concurrency)
        semaphore = asyncio.Semaphore(concurrency)

        async def _run_one(message_id: str, payload: dict[str, str]) -> tuple[bool, bool]:
            async with semaphore:
                return await self._process_message(message_id, payload)

        outcomes = await asyncio.gather(*(_run_one(message_id, payload) for message_id, payload in batch), return_exceptions=False)
        for processed, dead_lettered in outcomes:
            if processed:
                result.processed += 1
            if dead_lettered:
                result.dead_lettered += 1
            elif not processed:
                result.retried += 1
        return result

    async def run(self, *, stop_event: asyncio.Event | None = None) -> None:
        stop_event = stop_event or asyncio.Event()
        while not stop_event.is_set():
            await self.consume_batch()