from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from typing import Any

from redis.asyncio import Redis

from app.config import settings
from app.redis_client import normalized_event_fingerprint, publish_json


logger = logging.getLogger(__name__)


def _event_value(event: Any, field_name: str, default: str = "") -> str:
    if isinstance(event, dict):
        return str(event.get(field_name, default) or default)
    return str(getattr(event, field_name, default) or default)


def _event_as_dict(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return dict(event)
    if is_dataclass(event):
        return asdict(event)
    return {
        "source": _event_value(event, "source"),
        "source_event_id": _event_value(event, "source_event_id"),
        "title": _event_value(event, "title"),
        "headline": _event_value(event, "headline"),
        "body": _event_value(event, "body"),
        "url": _event_value(event, "url"),
        "published_at": _event_value(event, "published_at"),
        "tickers": list(getattr(event, "tickers", [])),
        "company_name": _event_value(event, "company_name"),
    }


class RedisStreamProducer:
    def __init__(self, redis: Redis, *, stream_name: str | None = None) -> None:
        self.redis = redis
        self.stream_name = stream_name or settings.redis_stream

    async def publish(self, event: Any) -> str | None:
        normalized = _event_as_dict(event)
        fingerprint = normalized_event_fingerprint(
            event if hasattr(event, "source") or isinstance(event, dict) else normalized
        )
        dedupe_key = f"news:producer:idempotency:{fingerprint}"
        accepted = await self.redis.set(
            dedupe_key,
            normalized.get("source_event_id", ""),
            nx=True,
            ex=settings.redis_message_idempotency_ttl,
        )
        if not accepted:
            logger.info(
                "producer skipped duplicate event",
                extra={
                    "source": normalized.get("source"),
                    "source_event_id": normalized.get("source_event_id"),
                    "url": normalized.get("url"),
                },
            )
            return None

        ticker_value = normalized.get("ticker") or ""
        tickers_raw = normalized.get("tickers")
        if not ticker_value and tickers_raw:
            if isinstance(tickers_raw, str):
                try:
                    parsed_tickers = json.loads(tickers_raw)
                except json.JSONDecodeError:
                    parsed_tickers = [tickers_raw]
                ticker_value = str(parsed_tickers[0]) if parsed_tickers else ""
            elif isinstance(tickers_raw, (list, tuple)) and tickers_raw:
                ticker_value = str(tickers_raw[0])

        tickers_json = tickers_raw if isinstance(tickers_raw, str) else json.dumps(list(tickers_raw or []))

        payload = {
            "source": normalized.get("source") or "",
            "source_event_id": normalized.get("source_event_id") or "",
            "headline": normalized.get("headline") or normalized.get("title") or "",
            "body": normalized.get("body") or "",
            "ticker": ticker_value,
            "tickers": tickers_json,
            "published_at": normalized.get("published_at") or "",
            "url": normalized.get("url") or "",
            "company_name": normalized.get("company_name") or "",
        }
        stream_id = await publish_json(self.redis, self.stream_name, payload)
        if stream_id is None:
            return None

        logger.info(
            "published event to redis stream",
            extra={
                "stream": self.stream_name,
                "stream_id": stream_id,
                "source": payload["source"],
                "source_event_id": payload["source_event_id"],
            },
        )
        return stream_id
