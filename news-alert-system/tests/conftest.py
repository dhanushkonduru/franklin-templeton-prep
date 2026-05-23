from __future__ import annotations

import pytest


@pytest.fixture()
def sample_raw_event():
    from datetime import datetime, timezone

    from app.types import RawNewsEvent

    return RawNewsEvent(
        source="newsapi",
        source_event_id="event-1",
        title="Apple reports strong earnings and raises guidance",
        body="Apple beat revenue estimates and raised guidance for the full year.",
        url="https://example.com/apple-earnings",
        published_at=datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc),
        tickers=("AAPL",),
        company_name="Apple Inc.",
        raw_payload={"example": True},
    )
