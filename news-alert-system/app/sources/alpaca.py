from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import aiohttp

from app.config import settings
from app.sources.base import BaseSource
from app.types import RawNewsEvent
from app.utils.http import fetch_json


class AlpacaNewsSource(BaseSource):
    name = "alpaca"

    def __init__(self, api_key: str | None = None, api_secret: str | None = None, session: aiohttp.ClientSession | None = None) -> None:
        self.api_key = api_key or settings.alpaca_api_key
        self.api_secret = api_secret or settings.alpaca_api_secret
        self.session = session

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    async def fetch_events(self) -> list[RawNewsEvent]:
        if not self.api_key or not self.api_secret:
            return []
        session = self.session or aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=settings.http_timeout_seconds))
        close_session = self.session is None
        try:
            payload = await fetch_json(
                session,
                settings.alpaca_news_url,
                headers=self._headers(),
                params={"limit": 50, "sort": "desc"},
                timeout_seconds=settings.http_timeout_seconds,
                attempts=settings.max_fetch_retries,
                operation_name="alpaca_news_fetch",
            )
            articles = list(payload.get("news", []))
        finally:
            if close_session:
                await session.close()
        events: list[RawNewsEvent] = []
        for article in articles:
            published_at = datetime.fromisoformat(article.get("created_at", datetime.now(timezone.utc).isoformat()).replace("Z", "+00:00"))
            symbols = tuple(str(symbol).upper() for symbol in article.get("symbols", []) if str(symbol).strip())
            source_id = article.get("id") or article.get("url") or article.get("headline", "")
            events.append(
                RawNewsEvent(
                    source=self.name,
                    source_event_id=str(source_id),
                    title=article.get("headline") or article.get("title") or "",
                    body=article.get("summary") or article.get("content") or "",
                    url=article.get("url") or "",
                    published_at=published_at,
                    tickers=symbols,
                    company_name=article.get("symbols", [None])[0] if article.get("symbols") else None,
                    raw_payload=article,
                )
            )
        return events
