from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import aiohttp

from app.config import settings
from app.sources.base import BaseSource
from app.types import RawNewsEvent
from app.utils.http import fetch_json


class NewsApiSource(BaseSource):
    name = "newsapi"

    def __init__(self, api_key: str | None = None, session: aiohttp.ClientSession | None = None) -> None:
        self.api_key = api_key or settings.newsapi_key
        self.session = session

    def _headers(self) -> dict[str, str]:
        return {"X-Api-Key": self.api_key}

    def _params(self) -> dict[str, Any]:
        return {
            "language": "en",
            "pageSize": 100,
            "sortBy": "publishedAt",
        }

    async def fetch_events(self) -> list[RawNewsEvent]:
        if not self.api_key:
            return []
        session = self.session or aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=settings.http_timeout_seconds))
        close_session = self.session is None
        try:
            payload = await fetch_json(
                session,
                settings.newsapi_url,
                headers=self._headers(),
                params=self._params(),
                timeout_seconds=settings.http_timeout_seconds,
                attempts=settings.max_fetch_retries,
                operation_name="newsapi_fetch",
            )
            articles = list(payload.get("articles", []))
        finally:
            if close_session:
                await session.close()
        events: list[RawNewsEvent] = []
        for article in articles:
            published_at = datetime.fromisoformat(article.get("publishedAt", datetime.now(timezone.utc).isoformat()).replace("Z", "+00:00"))
            title = article.get("title") or ""
            url = article.get("url") or ""
            source_id = article.get("url") or f"newsapi-{published_at.timestamp()}-{title[:64]}"
            events.append(
                RawNewsEvent(
                    source=self.name,
                    source_event_id=source_id,
                    title=title,
                    body=article.get("description") or article.get("content") or "",
                    url=url,
                    published_at=published_at,
                    tickers=tuple(),
                    company_name=article.get("source", {}).get("name"),
                    raw_payload=article,
                )
            )
        return events
