from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import timezone
from hashlib import sha256
from typing import TypedDict

import aiohttp
from redis.asyncio import Redis

from app.config import settings
from app.streams.producer import RedisStreamProducer
from app.sources.alpaca import AlpacaNewsSource
from app.sources.base import BaseSource
from app.sources.newsapi import NewsApiSource
from app.sources.sec_edgar import SecEdgarRssSource
from app.types import RawNewsEvent
from app.utils.text import normalize_text, normalize_ticker


logger = logging.getLogger(__name__)


class StreamNewsPayload(TypedDict):
    source: str
    source_event_id: str
    headline: str
    body: str
    ticker: str
    tickers: str
    published_at: str
    url: str
    company_name: str


@dataclass(slots=True)
class IngestionStats:
    fetched: int = 0
    published: int = 0
    duplicates: int = 0
    failed: int = 0


class AsyncNewsIngestionLayer:
    def __init__(
        self,
        redis: Redis,
        *,
        sources: list[BaseSource] | None = None,
        poll_interval_seconds: float | None = None,
        dedupe_ttl_seconds: int | None = None,
        http_timeout_seconds: float | None = None,
    ) -> None:
        self.redis = redis
        self.poll_interval_seconds = poll_interval_seconds or settings.news_poll_interval_seconds
        self.dedupe_ttl_seconds = dedupe_ttl_seconds or settings.dedup_ttl_seconds
        self.http_timeout_seconds = http_timeout_seconds or settings.http_timeout_seconds
        self.sources = sources or []
        self.producer = RedisStreamProducer(redis)

    def build_sources(self, session: aiohttp.ClientSession) -> list[BaseSource]:
        if self.sources:
            return self.sources
        return [
            NewsApiSource(session=session),
            SecEdgarRssSource(session=session),
            AlpacaNewsSource(session=session),
        ]

    def normalize(self, event: RawNewsEvent) -> StreamNewsPayload:
        headline = normalize_text(event.title)
        ticker_candidates = [normalize_ticker(ticker) for ticker in event.tickers if str(ticker).strip()]
        ticker = ticker_candidates[0] if ticker_candidates else ""
        published_at = event.published_at.astimezone(timezone.utc).isoformat()
        return {
            "source": event.source,
            "source_event_id": event.source_event_id,
            "headline": headline,
            "body": normalize_text(event.body),
            "ticker": ticker,
            "tickers": json.dumps(ticker_candidates),
            "published_at": published_at,
            "url": event.url,
            "company_name": normalize_text(event.company_name or ""),
        }

    def _dedupe_key(self, payload: StreamNewsPayload) -> str:
        digest = sha256()
        digest.update(payload["source"].encode("utf-8"))
        digest.update(b"|")
        digest.update(payload["source_event_id"].encode("utf-8"))
        digest.update(b"|")
        digest.update(payload["headline"].encode("utf-8"))
        digest.update(b"|")
        digest.update(payload["url"].encode("utf-8"))
        return f"news:ingest:dedupe:{digest.hexdigest()}"

    async def _should_publish(self, payload: StreamNewsPayload) -> bool:
        key = self._dedupe_key(payload)
        accepted = await self.redis.set(key, "1", ex=self.dedupe_ttl_seconds, nx=True)
        return bool(accepted)

    async def publish(self, payload: StreamNewsPayload) -> str | None:
        if not await self._should_publish(payload):
            logger.info(
                "duplicate skipped before redis",
                extra={"source": payload["source"], "ticker": payload["ticker"], "url": payload["url"]},
            )
            return None
        stream_id = await self.producer.publish(payload)
        logger.info(
            "published normalized event",
            extra={"source": payload["source"], "ticker": payload["ticker"], "stream_id": stream_id},
        )
        return stream_id

    async def publish_normalized_event(self, event: RawNewsEvent) -> str | None:
        normalized_payload = self.normalize(event)
        return await self.publish(normalized_payload)

    async def poll_once(self) -> IngestionStats:
        stats = IngestionStats()
        timeout = aiohttp.ClientTimeout(total=self.http_timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            sources = self.build_sources(session)
            for source in sources:
                try:
                    events = await source.fetch_events()
                    stats.fetched += len(events)
                    for event in events:
                        payload = self.normalize(event)
                        stream_id = await self.publish(payload)
                        if stream_id is None:
                            stats.duplicates += 1
                        else:
                            stats.published += 1
                except Exception as exc:
                    stats.failed += 1
                    logger.exception(
                        "ingestion source failed",
                        extra={"source": getattr(source, "name", "unknown"), "error": str(exc)},
                    )
        return stats

    async def run(self, *, stop_event: asyncio.Event | None = None) -> None:
        stop_event = stop_event or asyncio.Event()
        while not stop_event.is_set():
            try:
                await self.poll_once()
            except Exception as exc:
                logger.exception("ingestion poll failed, recovering", extra={"error": str(exc)})
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.poll_interval_seconds)
            except TimeoutError:
                continue
