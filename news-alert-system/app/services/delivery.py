from __future__ import annotations

import asyncio
import logging
import random
import smtplib
import time
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Any

import aiohttp
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import NotificationRecord, PortfolioHolding
from app.redis_client import publish_json
from app.types import NormalizedNewsEvent, utc_now


logger = logging.getLogger(__name__)


def _to_utc_iso(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _compute_latency_ms(event: NormalizedNewsEvent) -> int:
    return int(max(0.0, (utc_now() - event.published_at).total_seconds() * 1000.0))


def build_alert_payload(event: NormalizedNewsEvent, event_type: str, confidence: float) -> dict[str, str]:
    ticker = event.tickers[0] if event.tickers else ""
    return {
        "ticker": str(ticker),
        "headline": event.title,
        "event_type": event_type,
        "confidence": f"{confidence:.4f}",
        "published_time": _to_utc_iso(event.published_at),
        "latency_ms": str(_compute_latency_ms(event)),
    }


def build_slack_payload(alert_payload: dict[str, str]) -> dict[str, Any]:
    ticker = alert_payload.get("ticker", "") or "N/A"
    headline = alert_payload.get("headline", "")
    event_type = alert_payload.get("event_type", "")
    confidence = alert_payload.get("confidence", "")
    published_time = alert_payload.get("published_time", "")
    latency_ms = alert_payload.get("latency_ms", "")
    return {
        "text": f"News Alert: {ticker} - {headline}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*News Alert*\n*Ticker:* {ticker}\n*Headline:* {headline}",
                },
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"event_type={event_type}"},
                    {"type": "mrkdwn", "text": f"confidence={confidence}"},
                    {"type": "mrkdwn", "text": f"published_time={published_time}"},
                    {"type": "mrkdwn", "text": f"latency_ms={latency_ms}"},
                ],
            },
        ],
    }


def build_email_body(alert_payload: dict[str, str]) -> str:
    return (
        f"ticker: {alert_payload['ticker']}\n"
        f"headline: {alert_payload['headline']}\n"
        f"event_type: {alert_payload['event_type']}\n"
        f"confidence: {alert_payload['confidence']}\n"
        f"published_time: {alert_payload['published_time']}\n"
        f"latency_ms: {alert_payload['latency_ms']}\n"
    )


class RetryableDeliveryError(Exception):
    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class NonRetryableDeliveryError(Exception):
    pass


class AsyncRateLimiter:
    def __init__(
        self,
        rate_per_second: float,
        *,
        clock: Any | None = None,
        sleeper: Any | None = None,
    ) -> None:
        self.rate_per_second = max(0.0, float(rate_per_second))
        self._min_interval = 0.0 if self.rate_per_second <= 0 else 1.0 / self.rate_per_second
        self._clock = clock or time.monotonic
        self._sleeper = sleeper or asyncio.sleep
        self._lock = asyncio.Lock()
        self._next_available_at = 0.0

    async def wait(self) -> None:
        if self._min_interval <= 0:
            return

        async with self._lock:
            now = float(self._clock())
            if now < self._next_available_at:
                delay = self._next_available_at - now
                await self._sleeper(delay)
                now = float(self._clock())
            self._next_available_at = max(now, self._next_available_at) + self._min_interval


@dataclass(slots=True)
class DeliveryResult:
    attempts: int


class BaseDeliveryService:
    def __init__(
        self,
        *,
        rate_limiter: AsyncRateLimiter,
        timeout_seconds: float,
        retry_attempts: int,
        retry_base_delay_seconds: float,
        retry_max_delay_seconds: float,
    ) -> None:
        self.rate_limiter = rate_limiter
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = max(1, retry_attempts)
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.retry_max_delay_seconds = retry_max_delay_seconds

    async def _run_with_retry(self, operation: Any, *, operation_name: str) -> DeliveryResult:
        for attempt in range(1, self.retry_attempts + 1):
            await self.rate_limiter.wait()
            try:
                await operation()
                return DeliveryResult(attempts=attempt)
            except NonRetryableDeliveryError:
                raise
            except Exception as exc:
                if attempt >= self.retry_attempts:
                    raise

                retry_after = getattr(exc, "retry_after_seconds", None)
                if retry_after is not None:
                    delay = float(retry_after)
                else:
                    delay = min(self.retry_max_delay_seconds, self.retry_base_delay_seconds * (2 ** (attempt - 1)))
                delay = delay + random.uniform(0.0, max(delay * 0.15, 0.0))

                logger.warning(
                    "delivery retry scheduled",
                    extra={
                        "operation": operation_name,
                        "attempt": attempt,
                        "attempts": self.retry_attempts,
                        "delay_seconds": round(delay, 3),
                        "error": str(exc),
                    },
                )
                await asyncio.sleep(delay)

        return DeliveryResult(attempts=self.retry_attempts)


class WebhookDeliveryService(BaseDeliveryService):
    def __init__(self, timeout_seconds: float | None = None, rate_limiter: AsyncRateLimiter | None = None) -> None:
        super().__init__(
            rate_limiter=rate_limiter or AsyncRateLimiter(settings.delivery_webhook_rate_limit_per_second),
            timeout_seconds=timeout_seconds or settings.webhook_timeout_seconds,
            retry_attempts=settings.delivery_retry_attempts,
            retry_base_delay_seconds=settings.delivery_retry_base_delay_seconds,
            retry_max_delay_seconds=settings.delivery_retry_max_delay_seconds,
        )

    async def send(self, webhook_url: str, payload: dict[str, Any]) -> DeliveryResult:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:

            async def _post() -> str:
                async with session.post(webhook_url, json=payload) as response:
                    text = await response.text()
                    if response.status == 429 or 500 <= response.status < 600:
                        retry_after_value = response.headers.get("Retry-After")
                        retry_after_seconds = float(retry_after_value) if retry_after_value and retry_after_value.isdigit() else None
                        raise RetryableDeliveryError(
                            f"webhook server error status={response.status} body={text[:500]}",
                            retry_after_seconds=retry_after_seconds,
                        )
                    if response.status >= 400:
                        raise NonRetryableDeliveryError(f"webhook client error status={response.status} body={text[:500]}")
                    return text

            return await self._run_with_retry(_post, operation_name=f"webhook:{webhook_url}")


class SlackWebhookDeliveryService(WebhookDeliveryService):
    def __init__(self, timeout_seconds: float | None = None, rate_limiter: AsyncRateLimiter | None = None) -> None:
        super().__init__(timeout_seconds=timeout_seconds, rate_limiter=rate_limiter or AsyncRateLimiter(settings.delivery_slack_rate_limit_per_second))


class EmailDeliveryService(BaseDeliveryService):
    def __init__(self) -> None:
        super().__init__(
            rate_limiter=AsyncRateLimiter(settings.delivery_email_rate_limit_per_second),
            timeout_seconds=settings.webhook_timeout_seconds,
            retry_attempts=settings.delivery_retry_attempts,
            retry_base_delay_seconds=settings.delivery_retry_base_delay_seconds,
            retry_max_delay_seconds=settings.delivery_retry_max_delay_seconds,
        )
        self.host = settings.smtp_host
        self.port = settings.smtp_port
        self.username = settings.smtp_username
        self.password = settings.smtp_password
        self.from_address = settings.smtp_from_address
        self.use_tls = settings.smtp_use_tls

    def _build_message(self, to_address: str, subject: str, body: str) -> EmailMessage:
        message = EmailMessage()
        message["From"] = self.from_address
        message["To"] = to_address
        message["Subject"] = subject
        message.set_content(body)
        return message

    async def send(self, to_address: str, subject: str, body: str) -> DeliveryResult:
        message = self._build_message(to_address, subject, body)

        def _send_sync() -> None:
            with smtplib.SMTP(self.host, self.port, timeout=int(self.timeout_seconds)) as client:
                if self.use_tls:
                    client.starttls()
                if self.username:
                    client.login(self.username, self.password)
                client.send_message(message)

        async def _send() -> None:
            await asyncio.to_thread(_send_sync)

        return await self._run_with_retry(_send, operation_name=f"email:{to_address}")


class DeliveryFailureQueue:
    def __init__(self, redis: Redis | None = None, failure_stream: str | None = None) -> None:
        self.redis = redis
        self.failure_stream = failure_stream or settings.delivery_failure_stream

    async def enqueue(self, payload: dict[str, Any]) -> str | None:
        if self.redis is None:
            logger.error("delivery failure queue unavailable", extra={"stream": self.failure_stream})
            return None
        return await publish_json(self.redis, self.failure_stream, payload)


class NotificationService:
    def __init__(
        self,
        *,
        redis: Redis | None = None,
        webhook_service: WebhookDeliveryService | None = None,
        email_service: EmailDeliveryService | None = None,
        slack_service: SlackWebhookDeliveryService | None = None,
        failure_queue: DeliveryFailureQueue | None = None,
    ) -> None:
        self.webhook_service = webhook_service or WebhookDeliveryService()
        self.email_service = email_service or EmailDeliveryService()
        self.slack_service = slack_service or SlackWebhookDeliveryService()
        self.failure_queue = failure_queue or DeliveryFailureQueue(redis=redis)

    async def _record_failure(
        self,
        *,
        event_id: str,
        portfolio_name: str,
        channel: str,
        destination: str,
        payload: dict[str, str],
        error_message: str,
    ) -> str | None:
        return await self.failure_queue.enqueue(
            {
                "event_id": event_id,
                "portfolio_name": portfolio_name,
                "channel": channel,
                "destination": destination,
                "payload": payload,
                "error": error_message,
                "failed_at": _to_utc_iso(utc_now()),
            }
        )

    async def _deliver_channel(
        self,
        *,
        event_id: str,
        portfolio_name: str,
        channel: str,
        destination: str,
        payload: dict[str, str],
        sender: Any,
    ) -> NotificationRecord:
        start_time = time.perf_counter()
        try:
            result: DeliveryResult = await sender()
            delivery_latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "delivery succeeded",
                extra={
                    "event_id": event_id,
                    "portfolio_name": portfolio_name,
                    "channel": channel,
                    "destination": destination,
                    "status": "sent",
                    "attempts": result.attempts,
                    "delivery_latency_ms": round(delivery_latency_ms, 2),
                },
            )
            return NotificationRecord(
                event_id=event_id,
                portfolio_name=portfolio_name,
                channel=channel,
                destination=destination,
                status="sent",
                attempts=result.attempts,
                latency_ms=delivery_latency_ms,
            )
        except Exception as exc:
            delivery_latency_ms = (time.perf_counter() - start_time) * 1000.0
            queue_id = await self._record_failure(
                event_id=event_id,
                portfolio_name=portfolio_name,
                channel=channel,
                destination=destination,
                payload=payload,
                error_message=str(exc),
            )
            status = "queued_failure" if queue_id else "failed"
            final_error = str(exc) if queue_id is None else f"{exc} | queue_id={queue_id}"
            logger.exception(
                "delivery failed",
                extra={
                    "event_id": event_id,
                    "portfolio_name": portfolio_name,
                    "channel": channel,
                    "destination": destination,
                    "status": status,
                    "delivery_latency_ms": round(delivery_latency_ms, 2),
                    "error": str(exc),
                    "queue_id": queue_id,
                },
            )
            return NotificationRecord(
                event_id=event_id,
                portfolio_name=portfolio_name,
                channel=channel,
                destination=destination,
                status=status,
                attempts=settings.delivery_retry_attempts,
                latency_ms=delivery_latency_ms,
                error_message=final_error,
            )

    async def dispatch(
        self,
        session: AsyncSession,
        event_id: str,
        event: NormalizedNewsEvent,
        event_type: str,
        matched_holdings: list[PortfolioHolding],
        similarity_score: float,
        confidence: float = 0.0,
    ) -> list[NotificationRecord]:
        del similarity_score
        records: list[NotificationRecord] = []
        payload = build_alert_payload(event, event_type=event_type, confidence=confidence)

        for holding in matched_holdings:
            if holding.webhook_url:
                webhook_record = await self._deliver_channel(
                    event_id=event_id,
                    portfolio_name=holding.portfolio_name,
                    channel="webhook",
                    destination=holding.webhook_url,
                    payload=payload,
                    sender=lambda: self.webhook_service.send(holding.webhook_url or "", payload),
                )
                records.append(webhook_record)

            if holding.slack_webhook_url:
                slack_payload = build_slack_payload(payload)
                slack_record = await self._deliver_channel(
                    event_id=event_id,
                    portfolio_name=holding.portfolio_name,
                    channel="slack_webhook",
                    destination=holding.slack_webhook_url,
                    payload=payload,
                    sender=lambda: self.slack_service.send(holding.slack_webhook_url or "", slack_payload),
                )
                records.append(slack_record)

            if holding.email_address:
                subject = f"Alert: {payload['event_type'].replace('_', ' ').title()} | {payload['ticker'] or 'N/A'}"
                body = build_email_body(payload)
                email_record = await self._deliver_channel(
                    event_id=event_id,
                    portfolio_name=holding.portfolio_name,
                    channel="email",
                    destination=holding.email_address,
                    payload=payload,
                    sender=lambda: self.email_service.send(holding.email_address or "", subject, body),
                )
                records.append(email_record)

        if records:
            session.add_all(records)
            await session.commit()
        return records
