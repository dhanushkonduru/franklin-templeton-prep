from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.config import settings
from app.db import engine, init_db
from app.redis_client import create_redis_client, ensure_stream_groups


logger = logging.getLogger(__name__)


@asynccontextmanager
async def application_lifespan(_: FastAPI) -> AsyncIterator[None]:
    redis = create_redis_client()
    try:
        await ensure_stream_groups(
            redis,
            stream=settings.redis_stream,
            dlq_stream=settings.redis_dlq_stream,
            group=settings.redis_consumer_group,
        )
        await init_db()
        logger.info("application startup complete", extra={"environment": settings.environment})
        yield
    finally:
        logger.info("application shutdown started")
        try:
            await redis.aclose()
        except Exception as exc:
            logger.error("failed to close redis during shutdown", exc_info=exc)
        try:
            await engine.dispose()
        except Exception as exc:
            logger.error("failed to dispose database engine during shutdown", exc_info=exc)
        logger.info("application shutdown complete")
