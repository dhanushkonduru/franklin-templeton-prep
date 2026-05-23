from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, status
from redis.asyncio import Redis
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_session
from app.ingestion import AsyncNewsIngestionLayer
from app.models import AlertRecord, DeduplicatedEventRecord, LatencyMetricRecord, NewsEventRecord
from app.repositories import AlertRepository
from app.redis_client import get_redis_client
from app.schemas import (
    HealthResponse,
    IngestEventRequest,
    IngestedEventResponse,
    NewsEventRead,
    PortfolioAliasCreate,
    PortfolioAliasRead,
    PortfolioHoldingCreate,
    PortfolioHoldingRead,
    PortfolioMatchRequest,
    PortfolioMatchResponse,
    PortfolioSubsidiaryCreate,
    PortfolioSubsidiaryRead,
)
from app.services.portfolio_matching import PortfolioMatchingService
from app.types import RawNewsEvent, utc_now


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.app_name, time=utc_now())


@router.get("/health/ready", response_model=HealthResponse)
async def readiness(
    session: AsyncSession = Depends(get_session),
    redis: Redis = Depends(get_redis_client),
) -> HealthResponse:
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        logger.error("readiness check failed for database", exc_info=exc)
        return HealthResponse(status="database_down", service=settings.app_name, time=utc_now())

    try:
        await redis.ping()
    except Exception as exc:
        logger.error("readiness check failed for redis", exc_info=exc)
        return HealthResponse(status="redis_down", service=settings.app_name, time=utc_now())

    return HealthResponse(status="ok", service=settings.app_name, time=utc_now())


@router.get("/health", response_model=HealthResponse)
async def health(
    session: AsyncSession = Depends(get_session),
    redis: Redis = Depends(get_redis_client),
) -> HealthResponse:
    return await readiness(session=session, redis=redis)


@router.get("/metrics")
async def metrics(session: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    try:
        total_events = int(await session.scalar(select(func.count(NewsEventRecord.id))) or 0)
        total_alerts = int(await session.scalar(select(func.count(AlertRecord.id))) or 0)

        total_dedup = int(await session.scalar(select(func.count(DeduplicatedEventRecord.id))) or 0)
        duplicate_events = int(
            await session.scalar(
                select(func.count(DeduplicatedEventRecord.id)).where(DeduplicatedEventRecord.is_duplicate.is_(True))
            )
            or 0
        )

        duplicate_ratio = (duplicate_events / total_dedup) if total_dedup > 0 else 0.0

        latency_query = select(
            LatencyMetricRecord.stage,
            func.avg(LatencyMetricRecord.processing_latency),
        ).group_by(LatencyMetricRecord.stage)

        latency_results = await session.execute(latency_query)
        stage_latencies = {stage: round(float(avg_val or 0.0) * 1000, 2) for stage, avg_val in latency_results.all()}

        return {
            "service": settings.app_name,
            "environment": settings.environment,
            "pipeline_metrics": {
                "total_events_processed": total_events,
                "total_alerts_dispatched": total_alerts,
                "deduplication": {
                    "total_checked": total_dedup,
                    "duplicates_found": duplicate_events,
                    "duplicate_ratio_percent": round(duplicate_ratio * 100, 2),
                },
            },
            "average_stage_latencies_ms": stage_latencies,
        }
    except Exception as exc:
        logger.error("failed to retrieve metrics", exc_info=exc)
        return {"error": "failed to retrieve metrics", "details": str(exc)}


@router.post("/events/ingest", response_model=IngestedEventResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_event(
    payload: IngestEventRequest,
    redis: Redis = Depends(get_redis_client),
) -> IngestedEventResponse:
    event = RawNewsEvent(
        source=payload.source,
        source_event_id=payload.source_event_id,
        title=payload.title,
        body=payload.body,
        url=payload.url,
        published_at=payload.published_at,
        tickers=tuple(payload.tickers),
        company_name=payload.company_name,
        raw_payload=payload.raw_payload,
    )
    layer = AsyncNewsIngestionLayer(redis)
    stream_id = await layer.publish_normalized_event(event)
    return IngestedEventResponse(
        accepted=stream_id is not None,
        source=event.source,
        source_event_id=event.source_event_id,
        stream_id=stream_id,
    )


@router.post("/portfolio/holdings", response_model=PortfolioHoldingRead, status_code=status.HTTP_201_CREATED)
async def create_holding(payload: PortfolioHoldingCreate, session: AsyncSession = Depends(get_session)) -> PortfolioHoldingRead:
    repository = AlertRepository(session)
    record = await repository.add_holding(payload)
    return PortfolioHoldingRead.model_validate(record)


@router.get("/portfolio/holdings", response_model=list[PortfolioHoldingRead])
async def list_holdings(session: AsyncSession = Depends(get_session)) -> list[PortfolioHoldingRead]:
    repository = AlertRepository(session)
    holdings = await repository.list_holdings(active_only=False)
    return [PortfolioHoldingRead.model_validate(holding) for holding in holdings]


@router.get("/events", response_model=list[NewsEventRead])
async def list_events(limit: int = 100, session: AsyncSession = Depends(get_session)) -> list[NewsEventRead]:
    repository = AlertRepository(session)
    records = await repository.list_recent_events(limit=limit)
    return [NewsEventRead.model_validate(record) for record in records]


@router.post("/portfolio/aliases", response_model=PortfolioAliasRead, status_code=status.HTTP_201_CREATED)
async def create_alias(payload: PortfolioAliasCreate, session: AsyncSession = Depends(get_session)) -> PortfolioAliasRead:
    repository = AlertRepository(session)
    record = await repository.add_alias(payload)
    return PortfolioAliasRead.model_validate(record)


@router.post("/portfolio/subsidiaries", response_model=PortfolioSubsidiaryRead, status_code=status.HTTP_201_CREATED)
async def create_subsidiary(
    payload: PortfolioSubsidiaryCreate,
    session: AsyncSession = Depends(get_session),
) -> PortfolioSubsidiaryRead:
    repository = AlertRepository(session)
    record = await repository.add_subsidiary(payload)
    return PortfolioSubsidiaryRead.model_validate(record)


@router.post("/portfolio/match", response_model=PortfolioMatchResponse)
async def match_portfolio(
    payload: PortfolioMatchRequest,
    session: AsyncSession = Depends(get_session),
    redis: Redis = Depends(get_redis_client),
) -> PortfolioMatchResponse:
    service = PortfolioMatchingService(redis=redis, session=session)
    return await service.match(payload)


def create_app(*, lifespan=None) -> FastAPI:
    app = FastAPI(
        title="Real-Time News + Filing Alert System",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app
