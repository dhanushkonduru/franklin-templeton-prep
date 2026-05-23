from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from xml.etree import ElementTree as ET

import aiohttp

from app.config import settings
from app.sources.base import BaseSource
from app.types import RawNewsEvent
from app.utils.http import fetch_text


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class SecEdgarRssSource(BaseSource):
    name = "sec"

    def __init__(self, rss_url: str | None = None, session: aiohttp.ClientSession | None = None) -> None:
        self.rss_url = rss_url or settings.sec_rss_url
        self.session = session

    async def fetch_events(self) -> list[RawNewsEvent]:
        session = self.session or aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=settings.http_timeout_seconds), headers={"User-Agent": "news-alert-system/1.0 contact@example.com"})
        close_session = self.session is None
        try:
            xml_text = await fetch_text(
                session,
                self.rss_url,
                timeout_seconds=settings.http_timeout_seconds,
                attempts=settings.max_fetch_retries,
                operation_name="sec_rss_fetch",
            )
        finally:
            if close_session:
                await session.close()
        root = ET.fromstring(xml_text)
        events: list[RawNewsEvent] = []
        for entry in root.findall("atom:entry", ATOM_NS):
            title = entry.findtext("atom:title", default="", namespaces=ATOM_NS)
            link_element = entry.find("atom:link", ATOM_NS)
            url = link_element.attrib.get("href", "") if link_element is not None else ""
            updated = entry.findtext("atom:updated", default=datetime.now(timezone.utc).isoformat(), namespaces=ATOM_NS)
            published_at = parsedate_to_datetime(updated) if "GMT" in updated else datetime.fromisoformat(updated.replace("Z", "+00:00"))
            summary = entry.findtext("atom:summary", default="", namespaces=ATOM_NS)
            source_id = entry.findtext("atom:id", default=url, namespaces=ATOM_NS)
            events.append(
                RawNewsEvent(
                    source=self.name,
                    source_event_id=source_id,
                    title=title,
                    body=summary,
                    url=url,
                    published_at=published_at,
                    tickers=tuple(),
                    company_name=None,
                    raw_payload={"title": title, "url": url, "summary": summary},
                )
            )
        return events
