from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

import hashlib
from datetime import datetime

from app.models import (
    NewsEventRecord,
    NotificationRecord,
    PortfolioAliasRecord,
    PortfolioHolding,
    PortfolioMatchHistory,
    PortfolioSubsidiaryRecord,
    PortfolioWatchlist,
    AlertRecord,
    DeduplicatedEventRecord,
    LatencyMetricRecord,
)
from app.schemas import PortfolioAliasCreate, PortfolioHoldingCreate, PortfolioSubsidiaryCreate
from app.types import NormalizedNewsEvent
from app.utils.text import compute_embedding_hash


class AlertRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert_event(
        self,
        event: NormalizedNewsEvent,
        *,
        event_type: str,
        confidence: float,
        similarity_score: float,
        duplicate_of: str | None,
        embedding: list[float] | None = None,
        embedding_hash: str | None = None,
        processing_latency: float = 0.0,
    ) -> NewsEventRecord:
        emb_hash = embedding_hash or (compute_embedding_hash(embedding) if embedding else hashlib.sha256(event.title.encode("utf-8")).hexdigest())
        statement = select(NewsEventRecord).where(NewsEventRecord.source == event.source, NewsEventRecord.content_hash == event.content_hash)
        existing = await self.session.scalar(statement)
        if existing is not None:
            existing.event_type = event_type
            existing.event_confidence = confidence
            existing.similarity_score = similarity_score
            existing.duplicate_of = duplicate_of
            existing.raw_payload = event.raw_payload
            existing.headline = event.title
            existing.ticker = list(event.tickers)[0] if event.tickers else ""
            existing.embedding_hash = emb_hash
            existing.confidence = confidence
            existing.publication_time = event.published_at
            existing.processing_latency = processing_latency
            await self.session.commit()
            await self.session.refresh(existing)
            return existing

        record = NewsEventRecord(
            source=event.source,
            source_event_id=event.source_event_id,
            title=event.title,
            body=event.body,
            url=event.url,
            published_at=event.published_at,
            ticker_symbols=list(event.tickers),
            company_name=event.company_name,
            content_hash=event.content_hash,
            event_type=event_type,
            event_confidence=confidence,
            similarity_score=similarity_score,
            duplicate_of=duplicate_of,
            raw_payload=event.raw_payload,
            headline=event.title,
            ticker=list(event.tickers)[0] if event.tickers else "",
            embedding_hash=emb_hash,
            confidence=confidence,
            publication_time=event.published_at,
            processing_latency=processing_latency,
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def find_event_id_by_source_event_id(self, source: str, source_event_id: str) -> str | None:
        statement = select(NewsEventRecord.id).where(
            NewsEventRecord.source == source,
            NewsEventRecord.source_event_id == source_event_id,
        )
        return await self.session.scalar(statement)

    async def list_recent_events(self, limit: int = 100) -> list[NewsEventRecord]:
        statement = select(NewsEventRecord).order_by(NewsEventRecord.published_at.desc()).limit(limit)
        result = await self.session.scalars(statement)
        return list(result)

    async def add_holding(self, payload: PortfolioHoldingCreate) -> PortfolioHolding:
        existing = await self.session.scalar(
            select(PortfolioHolding).where(
                PortfolioHolding.portfolio_name == payload.portfolio_name,
                PortfolioHolding.ticker == payload.ticker,
            )
        )
        if existing is not None:
            existing.company_name = payload.company_name
            existing.active = payload.active
            existing.webhook_url = payload.webhook_url
            existing.slack_webhook_url = payload.slack_webhook_url
            existing.email_address = payload.email_address
            await self.session.commit()
            await self.session.refresh(existing)
            return existing

        holding = PortfolioHolding(
            portfolio_name=payload.portfolio_name,
            ticker=payload.ticker,
            company_name=payload.company_name,
            active=payload.active,
            webhook_url=payload.webhook_url,
            slack_webhook_url=payload.slack_webhook_url,
            email_address=payload.email_address,
        )
        self.session.add(holding)
        await self.session.commit()
        await self.session.refresh(holding)
        return holding

    async def list_holdings(self, active_only: bool = True) -> list[PortfolioHolding]:
        statement = select(PortfolioHolding)
        if active_only:
            statement = statement.where(PortfolioHolding.active.is_(True))
        statement = statement.order_by(PortfolioHolding.portfolio_name.asc(), PortfolioHolding.ticker.asc())
        result = await self.session.scalars(statement)
        return list(result)

    async def list_watchlists(self, active_only: bool = True) -> list[PortfolioWatchlist]:
        statement = select(PortfolioWatchlist)
        if active_only:
            statement = statement.where(PortfolioWatchlist.active.is_(True))
        statement = statement.order_by(PortfolioWatchlist.portfolio_name.asc())
        result = await self.session.scalars(statement)
        return list(result)

    async def add_alias(self, payload: PortfolioAliasCreate) -> PortfolioAliasRecord:
        existing = await self.session.scalar(
            select(PortfolioAliasRecord).where(
                PortfolioAliasRecord.portfolio_symbol == payload.portfolio_symbol,
                PortfolioAliasRecord.alias == payload.alias,
            )
        )
        if existing is not None:
            existing.active = payload.active
            await self.session.commit()
            await self.session.refresh(existing)
            return existing

        record = PortfolioAliasRecord(
            portfolio_symbol=payload.portfolio_symbol,
            alias=payload.alias,
            active=payload.active,
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def add_subsidiary(self, payload: PortfolioSubsidiaryCreate) -> PortfolioSubsidiaryRecord:
        existing = await self.session.scalar(
            select(PortfolioSubsidiaryRecord).where(
                PortfolioSubsidiaryRecord.portfolio_symbol == payload.portfolio_symbol,
                PortfolioSubsidiaryRecord.subsidiary_name == payload.subsidiary_name,
            )
        )
        if existing is not None:
            existing.active = payload.active
            await self.session.commit()
            await self.session.refresh(existing)
            return existing

        record = PortfolioSubsidiaryRecord(
            portfolio_symbol=payload.portfolio_symbol,
            subsidiary_name=payload.subsidiary_name,
            active=payload.active,
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def list_aliases(self, active_only: bool = True) -> list[PortfolioAliasRecord]:
        statement = select(PortfolioAliasRecord)
        if active_only:
            statement = statement.where(PortfolioAliasRecord.active.is_(True))
        result = await self.session.scalars(statement.order_by(PortfolioAliasRecord.portfolio_symbol.asc(), PortfolioAliasRecord.alias.asc()))
        return list(result)

    async def list_subsidiaries(self, active_only: bool = True) -> list[PortfolioSubsidiaryRecord]:
        statement = select(PortfolioSubsidiaryRecord)
        if active_only:
            statement = statement.where(PortfolioSubsidiaryRecord.active.is_(True))
        result = await self.session.scalars(statement.order_by(PortfolioSubsidiaryRecord.portfolio_symbol.asc(), PortfolioSubsidiaryRecord.subsidiary_name.asc()))
        return list(result)

    async def record_portfolio_match(self, record: PortfolioMatchHistory) -> PortfolioMatchHistory:
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def list_portfolio_match_history(self, limit: int = 100) -> list[PortfolioMatchHistory]:
        statement = select(PortfolioMatchHistory).order_by(PortfolioMatchHistory.created_at.desc()).limit(limit)
        result = await self.session.scalars(statement)
        return list(result)

    async def record_notification(self, record: NotificationRecord) -> NotificationRecord:
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def create_alert(
        self,
        *,
        headline: str,
        ticker: str,
        embedding: list[float] | None = None,
        embedding_hash: str | None = None,
        event_type: str,
        confidence: float,
        publication_time: datetime,
        processing_latency: float,
    ) -> AlertRecord:
        emb_hash = embedding_hash or (compute_embedding_hash(embedding) if embedding else hashlib.sha256(headline.encode("utf-8")).hexdigest())
        record = AlertRecord(
            headline=headline,
            ticker=ticker,
            embedding_hash=emb_hash,
            event_type=event_type,
            confidence=confidence,
            publication_time=publication_time,
            processing_latency=processing_latency,
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def list_alerts(self, *, ticker: str | None = None, limit: int = 100) -> list[AlertRecord]:
        statement = select(AlertRecord)
        if ticker:
            statement = statement.where(AlertRecord.ticker == ticker)
        statement = statement.order_by(AlertRecord.publication_time.desc()).limit(limit)
        result = await self.session.scalars(statement)
        return list(result)

    async def create_deduplicated_event(
        self,
        *,
        headline: str,
        ticker: str,
        embedding: list[float] | None = None,
        embedding_hash: str | None = None,
        event_type: str,
        confidence: float,
        publication_time: datetime,
        processing_latency: float,
        similarity_score: float,
        is_duplicate: bool,
        duplicate_of: str | None = None,
    ) -> DeduplicatedEventRecord:
        emb_hash = embedding_hash or (compute_embedding_hash(embedding) if embedding else hashlib.sha256(headline.encode("utf-8")).hexdigest())
        record = DeduplicatedEventRecord(
            headline=headline,
            ticker=ticker,
            embedding_hash=emb_hash,
            event_type=event_type,
            confidence=confidence,
            publication_time=publication_time,
            processing_latency=processing_latency,
            similarity_score=similarity_score,
            is_duplicate=is_duplicate,
            duplicate_of=duplicate_of,
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def list_deduplicated_events(self, *, ticker: str | None = None, limit: int = 100) -> list[DeduplicatedEventRecord]:
        statement = select(DeduplicatedEventRecord)
        if ticker:
            statement = statement.where(DeduplicatedEventRecord.ticker == ticker)
        statement = statement.order_by(DeduplicatedEventRecord.publication_time.desc()).limit(limit)
        result = await self.session.scalars(statement)
        return list(result)

    async def record_latency_metric(
        self,
        *,
        event_id: str,
        stage: str,
        headline: str,
        ticker: str,
        embedding: list[float] | None = None,
        embedding_hash: str | None = None,
        event_type: str,
        confidence: float,
        publication_time: datetime,
        processing_latency: float,
    ) -> LatencyMetricRecord:
        emb_hash = embedding_hash or (compute_embedding_hash(embedding) if embedding else hashlib.sha256(headline.encode("utf-8")).hexdigest())
        record = LatencyMetricRecord(
            event_id=event_id,
            stage=stage,
            headline=headline,
            ticker=ticker,
            embedding_hash=emb_hash,
            event_type=event_type,
            confidence=confidence,
            publication_time=publication_time,
            processing_latency=processing_latency,
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def list_latency_metrics(self, *, stage: str | None = None, limit: int = 100) -> list[LatencyMetricRecord]:
        statement = select(LatencyMetricRecord)
        if stage:
            statement = statement.where(LatencyMetricRecord.stage == stage)
        statement = statement.order_by(LatencyMetricRecord.created_at.desc()).limit(limit)
        result = await self.session.scalars(statement)
        return list(result)
