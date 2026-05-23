from __future__ import annotations

import asyncio
import logging
import signal
import sys

from app.config import settings
from app.db import async_session_factory, engine, init_db
from app.ingestion import AsyncNewsIngestionLayer
from app.logging_config import setup_logging
from app.pipeline.orchestrator import NewsPipeline
from app.redis_client import create_redis_client, ensure_stream_groups
from app.services.delivery_replay import DeliveryReplayConsumer
from app.streams.consumer import RedisStreamConsumer
from app.types import RawNewsEvent


setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


async def run_worker() -> None:
    logger.info("starting alert system worker", extra={"environment": settings.environment})
    redis = create_redis_client()

    try:
        await ensure_stream_groups(
            redis,
            stream=settings.redis_stream,
            dlq_stream=settings.redis_dlq_stream,
            group=settings.redis_consumer_group,
        )
        await ensure_stream_groups(
            redis,
            stream=settings.delivery_failure_stream,
            dlq_stream=f"{settings.delivery_failure_stream}:dlq",
            group=settings.delivery_replay_group,
        )
        await init_db()
    except Exception as exc:
        logger.critical("failed to initialize worker resources", exc_info=exc)
        await redis.aclose()
        await engine.dispose()
        sys.exit(1)

    pipeline = NewsPipeline(redis=redis, session_factory=async_session_factory)
    ingestion = AsyncNewsIngestionLayer(redis)
    replay_consumer = DeliveryReplayConsumer(redis, async_session_factory)

    async def process_raw_event(raw_event: RawNewsEvent) -> None:
        await pipeline.process_raw_event(raw_event)

    consumer = RedisStreamConsumer(redis, process_raw_event)
    stop_event = asyncio.Event()

    def handle_shutdown_signal() -> None:
        logger.info("received shutdown signal, stopping loops")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_shutdown_signal)
        except NotImplementedError:
            signal.signal(sig, lambda _signum, _frame: handle_shutdown_signal())

    async def consumer_loop() -> None:
        logger.info("consumer loop started")
        while not stop_event.is_set():
            try:
                await consumer.consume_batch()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("unhandled exception in consumer loop, recovering", exc_info=exc)
                await asyncio.sleep(settings.redis_consumer_retry_backoff_seconds)
        logger.info("consumer loop terminated")

    async def poller_loop() -> None:
        logger.info("poller loop started")
        try:
            await ingestion.run(stop_event=stop_event)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("unhandled exception in poller loop", exc_info=exc)
        logger.info("poller loop terminated")

    async def replay_loop() -> None:
        logger.info("delivery replay loop started")
        while not stop_event.is_set():
            try:
                await replay_consumer.consume_batch()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("unhandled exception in delivery replay loop, recovering", exc_info=exc)
                await asyncio.sleep(settings.redis_consumer_retry_backoff_seconds)
        logger.info("delivery replay loop terminated")

    try:
        async with asyncio.TaskGroup() as task_group:
            task_group.create_task(consumer_loop())
            task_group.create_task(poller_loop())
            task_group.create_task(replay_loop())
    except Exception as exc:
        logger.error("worker tasks encountered an exception", exc_info=exc)
    finally:
        logger.info("shutting down worker, cleaning up connections")
        try:
            await redis.aclose()
            logger.info("redis connection closed successfully")
        except Exception as exc:
            logger.error("error closing redis client", exc_info=exc)

        try:
            await engine.dispose()
            logger.info("database engine disposed successfully")
        except Exception as exc:
            logger.error("error disposing database engine", exc_info=exc)

        logger.info("worker shutdown complete")


def main() -> None:
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("worker terminated by keyboard interrupt")


if __name__ == "__main__":
    main()
