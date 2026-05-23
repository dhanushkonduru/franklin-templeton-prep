from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from typing import Any

from redis.asyncio import Redis

from app.config import settings


def create_redis_client(url: str | None = None) -> Redis:
    return Redis.from_url(url or settings.redis_url, decode_responses=True)


async def get_redis_client() -> Redis:
    return create_redis_client()


def normalized_event_fingerprint(event: Any) -> str:
    if isinstance(event, dict):
        source = str(event.get("source", ""))
        source_event_id = str(event.get("source_event_id", ""))
        content_hash = str(event.get("content_hash") or event.get("headline") or event.get("title") or "")
        url = str(event.get("url", ""))
    elif is_dataclass(event):
        payload = asdict(event)
        source = str(payload.get("source", ""))
        source_event_id = str(payload.get("source_event_id", ""))
        content_hash = str(payload.get("content_hash") or payload.get("headline") or payload.get("title") or "")
        url = str(payload.get("url", ""))
    else:
        source = str(getattr(event, "source", ""))
        source_event_id = str(getattr(event, "source_event_id", ""))
        content_hash = str(
            getattr(event, "content_hash", "") or getattr(event, "headline", "") or getattr(event, "title", "") or ""
        )
        url = str(getattr(event, "url", ""))

    digest = sha256()
    digest.update(source.encode("utf-8"))
    digest.update(b"|")
    digest.update(source_event_id.encode("utf-8"))
    digest.update(b"|")
    digest.update(content_hash.encode("utf-8"))
    digest.update(b"|")
    digest.update(url.encode("utf-8"))
    return digest.hexdigest()


async def ensure_stream_group(redis: Redis, stream: str, group: str) -> None:
    try:
        await redis.xgroup_create(name=stream, groupname=group, id="$", mkstream=True)
    except Exception as exc:  # pragma: no cover - exists on operational startup
        if "BUSYGROUP" not in str(exc):
            raise


async def ensure_stream_groups(
    redis: Redis,
    *,
    stream: str | None = None,
    dlq_stream: str | None = None,
    group: str | None = None,
) -> None:
    active_stream = stream or settings.redis_stream
    active_dlq = dlq_stream or settings.redis_dlq_stream
    active_group = group or settings.redis_consumer_group
    await ensure_stream_group(redis, active_stream, active_group)
    await ensure_stream_group(redis, active_dlq, f"{active_group}-dlq")


async def publish_json(redis: Redis, stream: str, payload: dict[str, Any]) -> str:
    serializable_payload = {
        key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value for key, value in payload.items()
    }
    return await redis.xadd(stream, serializable_payload, maxlen=10_000, approximate=True)


async def get_json(redis: Redis, key: str) -> dict[str, Any] | None:
    payload = await redis.get(key)
    if not payload:
        return None
    return json.loads(payload)


async def set_json(redis: Redis, key: str, payload: dict[str, Any], ttl_seconds: int) -> None:
    await redis.set(key, json.dumps(payload), ex=ttl_seconds)
