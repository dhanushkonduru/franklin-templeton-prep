from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class RawNewsEvent:
    source: str
    source_event_id: str
    title: str
    body: str
    url: str
    published_at: datetime
    tickers: tuple[str, ...] = field(default_factory=tuple)
    company_name: str | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedNewsEvent:
    source: str
    source_event_id: str
    title: str
    body: str
    url: str
    published_at: datetime
    tickers: tuple[str, ...]
    company_name: str | None
    content_hash: str
    raw_payload: dict[str, Any]
