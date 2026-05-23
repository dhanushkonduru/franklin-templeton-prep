from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from app.types import RawNewsEvent


def _serialize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    return value


def raw_event_to_dict(event: RawNewsEvent) -> dict[str, Any]:
    return {
        "source": event.source,
        "source_event_id": event.source_event_id,
        "title": event.title,
        "body": event.body,
        "url": event.url,
        "published_at": event.published_at.isoformat(),
        "tickers": list(event.tickers),
        "company_name": event.company_name,
        "raw_payload": event.raw_payload,
    }


def embedding_to_json(embedding: list[float]) -> str:
    return json.dumps(embedding, separators=(",", ":"))


def embedding_from_json(value: str) -> list[float]:
    data = json.loads(value)
    return [float(item) for item in data]
