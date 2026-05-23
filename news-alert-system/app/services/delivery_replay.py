from __future__ import annotations

import json
import logging

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import settings
from app.models import NotificationRecord
from app.redis_client import ensure_stream_group
from app.services.delivery import EmailDeliveryService, SlackWebhookDeliveryService, WebhookDeliveryService, build_email_body


logger = logging.getLogger(__name__)


class DeliveryReplayConsumer:
    def __init__(
        self,
        redis: Redis,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        failure_stream: str | None = None,
        group: str | None = None,
        consumer_name: str | None = None,
        batch_size: int | None = None,
        replay_max_attempts: int | None = None,
        webhook_service: WebhookDeliveryService | None = None,
        email_service: EmailDeliveryService | None = None,
        slack_service: SlackWebhookDeliveryService | None = None,
    ) -> None:
        self.redis = redis
        self.session_factory = session_factory
        self.failure_stream = failure_stream or settings.delivery_failure_stream
        self.group = group or settings.delivery_replay_group
        self.consumer_name = consumer_name or settings.delivery_replay_consumer_name
        self.batch_size = batch_size or settings.delivery_replay_batch_size
        self.replay_max_attempts = replay_max_attempts or settings.delivery_replay_max_attempts
        self.webhook_service = webhook_service or WebhookDeliveryService()
        self.email_service = email_service or EmailDeliveryService()
        self.slack_service = slack_service or SlackWebhookDeliveryService()

    async def ensure_group(self) -> None:
        await ensure_stream_group(self.redis, self.failure_stream, self.group)

    @staticmethod
    def _deserialize_payload(value: str | dict[str, str]) -> dict[str, str]:
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        try:
            payload = json.loads(value)
            if isinstance(payload, dict):
                return {str(k): str(v) for k, v in payload.items()}
        except Exception:
            pass
        return {}

    async def _send_once(self, channel: str, destination: str, payload: dict[str, str]) -> int:
        if channel == "webhook":
            result = await self.webhook_service.send(destination, payload)
            return result.attempts
        if channel == "slack_webhook":
            result = await self.slack_service.send(destination, payload)
            return result.attempts
        if channel == "email":
            subject = f"Alert: {payload.get('event_type', 'other').replace('_', ' ').title()} | {payload.get('ticker') or 'N/A'}"
            body = build_email_body(
                {
                    "ticker": payload.get("ticker", ""),
                    "headline": payload.get("headline", ""),
                    "event_type": payload.get("event_type", "other"),
                    "confidence": payload.get("confidence", "0.0000"),
                    "published_time": payload.get("published_time", ""),
                    "latency_ms": payload.get("latency_ms", "0"),
                }
            )
            result = await self.email_service.send(destination, subject, body)
            return result.attempts
        raise ValueError(f"unsupported replay channel: {channel}")

    async def _record_replay_status(
        self,
        *,
        event_id: str,
        portfolio_name: str,
        channel: str,
        destination: str,
        status: str,
        attempts: int,
        error_message: str | None,
    ) -> None:
        async with self.session_factory() as session:
            session.add(
                NotificationRecord(
                    event_id=event_id,
                    portfolio_name=portfolio_name,
                    channel=channel,
                    destination=destination,
                    status=status,
                    attempts=attempts,
                    error_message=error_message,
                )
            )
            await session.commit()

    async def _requeue(self, payload: dict[str, str], replay_attempts: int) -> str:
        return await self.redis.xadd(
            self.failure_stream,
            {
                **payload,
                "replay_attempts": str(replay_attempts),
            },
            maxlen=10_000,
            approximate=True,
        )

    async def _process_message(self, message_id: str, message: dict[str, str]) -> None:
        event_id = message.get("event_id", "")
        portfolio_name = message.get("portfolio_name", "")
        channel = message.get("channel", "")
        destination = message.get("destination", "")
        replay_attempts = int(message.get("replay_attempts", "0"))
        payload = self._deserialize_payload(message.get("payload", "{}"))

        try:
            delivery_attempts = await self._send_once(channel, destination, payload)
            await self.redis.xack(self.failure_stream, self.group, message_id)
            await self._record_replay_status(
                event_id=event_id,
                portfolio_name=portfolio_name,
                channel=channel,
                destination=destination,
                status="replayed_sent",
                attempts=delivery_attempts,
                error_message=None,
            )
            logger.info(
                "delivery replay succeeded",
                extra={
                    "event_id": event_id,
                    "portfolio_name": portfolio_name,
                    "channel": channel,
                    "destination": destination,
                    "message_id": message_id,
                },
            )
        except Exception as exc:
            next_attempt = replay_attempts + 1
            if next_attempt >= self.replay_max_attempts:
                await self.redis.xack(self.failure_stream, self.group, message_id)
                await self._record_replay_status(
                    event_id=event_id,
                    portfolio_name=portfolio_name,
                    channel=channel,
                    destination=destination,
                    status="replay_failed",
                    attempts=next_attempt,
                    error_message=str(exc),
                )
                logger.exception(
                    "delivery replay exhausted",
                    extra={
                        "event_id": event_id,
                        "portfolio_name": portfolio_name,
                        "channel": channel,
                        "destination": destination,
                        "message_id": message_id,
                        "attempts": next_attempt,
                    },
                )
                return

            requeue_payload = {k: str(v) for k, v in message.items() if k != "replay_attempts"}
            await self._requeue(requeue_payload, replay_attempts=next_attempt)
            await self.redis.xack(self.failure_stream, self.group, message_id)
            logger.warning(
                "delivery replay requeued",
                extra={
                    "event_id": event_id,
                    "portfolio_name": portfolio_name,
                    "channel": channel,
                    "destination": destination,
                    "message_id": message_id,
                    "attempts": next_attempt,
                    "error": str(exc),
                },
            )

    async def consume_batch(self) -> int:
        await self.ensure_group()
        messages = await self.redis.xreadgroup(
            groupname=self.group,
            consumername=self.consumer_name,
            streams={self.failure_stream: ">"},
            count=self.batch_size,
            block=1000,
        )

        batch: list[tuple[str, dict[str, str]]] = []
        for _, stream_messages in messages:
            batch.extend(stream_messages)

        if not batch:
            return 0

        for message_id, message_payload in batch:
            await self._process_message(message_id, message_payload)
        return len(batch)
