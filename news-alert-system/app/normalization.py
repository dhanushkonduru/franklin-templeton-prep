from __future__ import annotations

from app.streaming import raw_event_to_dict
from app.types import NormalizedNewsEvent, RawNewsEvent
from app.utils.text import content_hash, normalize_ticker


def normalize_event(event: RawNewsEvent) -> NormalizedNewsEvent:
    tickers = tuple(sorted({normalize_ticker(ticker) for ticker in event.tickers if ticker.strip()}))
    normalized_title = " ".join(event.title.split())
    normalized_body = " ".join(event.body.split())
    hash_value = content_hash(event.source, event.source_event_id, normalized_title, normalized_body, event.url)
    return NormalizedNewsEvent(
        source=event.source,
        source_event_id=event.source_event_id,
        title=normalized_title,
        body=normalized_body,
        url=event.url,
        published_at=event.published_at,
        tickers=tickers,
        company_name=event.company_name.strip() if event.company_name else None,
        content_hash=hash_value,
        raw_payload=raw_event_to_dict(event),
    )


def event_text(event: NormalizedNewsEvent) -> str:
    ticker_text = " ".join(event.tickers)
    company_text = event.company_name or ""
    return " ".join(part for part in [event.title, event.body, ticker_text, company_text] if part).strip()
