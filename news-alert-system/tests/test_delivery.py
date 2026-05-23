from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import func, select

from app.db import create_engine_and_sessionmaker, init_db
from app.models import NotificationRecord, PortfolioHolding
from app.services.delivery import AsyncRateLimiter, DeliveryResult, NotificationService, build_alert_payload
from app.types import NormalizedNewsEvent, utc_now


class FakeRedis:
    def __init__(self) -> None:
        self.messages: list[tuple[str, dict[str, object]]] = []

    async def xadd(self, stream: str, payload: dict[str, object], maxlen: int | None = None, approximate: bool = True) -> str:
        del maxlen
        del approximate
        self.messages.append((stream, payload))
        return f"{len(self.messages)}-0"


class AlwaysSuccessDelivery:
    async def send(self, *args: object, **kwargs: object) -> DeliveryResult:
        del args
        del kwargs
        return DeliveryResult(attempts=1)


class AlwaysFailDelivery:
    async def send(self, *args: object, **kwargs: object) -> DeliveryResult:
        del args
        del kwargs
        raise RuntimeError("delivery provider unavailable")


def _event() -> NormalizedNewsEvent:
    return NormalizedNewsEvent(
        source="newsapi",
        source_event_id="ev-1",
        title="Apple posts record revenue",
        body="Q1 results beat expectations",
        url="https://example.com/news/1",
        published_at=utc_now() - timedelta(seconds=2),
        tickers=("AAPL",),
        company_name="Apple Inc.",
        content_hash="hash-1",
        raw_payload={"id": "ev-1"},
    )


def test_build_alert_payload_contract():
    payload = build_alert_payload(_event(), event_type="earnings", confidence=0.93)

    assert set(payload.keys()) == {
        "ticker",
        "headline",
        "event_type",
        "confidence",
        "published_time",
        "latency_ms",
    }
    assert payload["ticker"] == "AAPL"
    assert payload["headline"] == "Apple posts record revenue"
    assert payload["event_type"] == "earnings"
    assert payload["confidence"] == "0.9300"
    assert payload["published_time"]
    assert int(payload["latency_ms"]) >= 0


@pytest.mark.asyncio
async def test_dispatch_queues_failed_delivery_and_tracks_status():
    engine, session_factory = create_engine_and_sessionmaker("sqlite+aiosqlite:///:memory:")
    await init_db(engine)

    holding = PortfolioHolding(
        portfolio_name="Core",
        ticker="AAPL",
        company_name="Apple Inc.",
        active=True,
        webhook_url="https://example.com/webhook",
        slack_webhook_url="https://hooks.slack.com/services/T/B/C",
        email_address="alerts@example.com",
    )

    redis = FakeRedis()
    service = NotificationService(
        redis=redis,
        webhook_service=AlwaysFailDelivery(),
        email_service=AlwaysSuccessDelivery(),
        slack_service=AlwaysSuccessDelivery(),
    )

    async with session_factory() as session:
        records = await service.dispatch(
            session=session,
            event_id="event-123",
            event=_event(),
            event_type="earnings",
            matched_holdings=[holding],
            similarity_score=0.95,
            confidence=0.88,
        )

        count_statement = select(func.count()).select_from(NotificationRecord)
        persisted_count = int((await session.scalar(count_statement)) or 0)

    assert len(records) == 3
    statuses = {record.channel: record.status for record in records}
    assert statuses["webhook"] == "queued_failure"
    assert statuses["slack_webhook"] == "sent"
    assert statuses["email"] == "sent"
    assert persisted_count == 3

    assert len(redis.messages) == 1
    stream, payload = redis.messages[0]
    assert stream == "notification_delivery_failures"
    assert payload["channel"] == "webhook"


@pytest.mark.asyncio
async def test_rate_limiter_enforces_delay_deterministically():
    now = 100.0
    sleep_calls: list[float] = []

    def fake_clock() -> float:
        return now

    async def fake_sleep(seconds: float) -> None:
        nonlocal now
        sleep_calls.append(seconds)
        now += seconds

    limiter = AsyncRateLimiter(rate_per_second=2.0, clock=fake_clock, sleeper=fake_sleep)

    await limiter.wait()
    await limiter.wait()

    assert len(sleep_calls) == 1
    assert sleep_calls[0] == pytest.approx(0.5, rel=1e-3)
