from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


def json_column() -> JSON | JSONB:
    return JSON().with_variant(JSONB, "postgresql")


class NewsEventRecord(Base):
    __tablename__ = "news_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    source: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    source_event_id: Mapped[str] = mapped_column(String(256), index=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, default="", nullable=False)
    url: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    ticker_symbols: Mapped[list[str]] = mapped_column(json_column(), default=list, nullable=False)
    company_name: Mapped[str | None] = mapped_column(String(255), index=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    event_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    duplicate_of: Mapped[str | None] = mapped_column(String(36), ForeignKey("news_events.id"), nullable=True)
    raw_payload: Mapped[dict] = mapped_column(json_column(), default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Added fields to satisfy requirements
    headline: Mapped[str] = mapped_column(Text, nullable=False, default="")
    ticker: Mapped[str] = mapped_column(String(16), index=True, nullable=False, default="")
    embedding_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False, default="")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    publication_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False, server_default=func.now())
    processing_latency: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    duplicate_parent: Mapped["NewsEventRecord | None"] = relationship(remote_side="NewsEventRecord.id", uselist=False)


class PortfolioHolding(Base):
    __tablename__ = "portfolio_holdings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    portfolio_name: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    ticker: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    company_name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    webhook_url: Mapped[str | None] = mapped_column(Text)
    slack_webhook_url: Mapped[str | None] = mapped_column(Text)
    email_address: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class PortfolioAliasRecord(Base):
    __tablename__ = "portfolio_aliases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    portfolio_symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    alias: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class PortfolioSubsidiaryRecord(Base):
    __tablename__ = "portfolio_subsidiaries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    portfolio_symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    subsidiary_name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class NotificationRecord(Base):
    __tablename__ = "notification_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    event_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    portfolio_name: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    channel: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    destination: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    attempts: Mapped[int] = mapped_column(nullable=False, default=1)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    error_message: Mapped[str | None] = mapped_column(Text)
    delivered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class PortfolioWatchlist(Base):
    __tablename__ = "portfolio_watchlists"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    portfolio_name: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    keywords: Mapped[list[str]] = mapped_column(json_column(), default=list, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class PortfolioMatchHistory(Base):
    __tablename__ = "portfolio_match_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, default="")
    event_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    portfolio_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    matched_symbol: Mapped[str | None] = mapped_column(String(16), index=True)
    match_reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_payload: Mapped[dict] = mapped_column(json_column(), default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


Index("ix_news_events_source_hash", NewsEventRecord.source, NewsEventRecord.content_hash, unique=True)
Index("ix_portfolio_holdings_portfolio_ticker", PortfolioHolding.portfolio_name, PortfolioHolding.ticker, unique=True)
Index("ix_portfolio_aliases_symbol_alias", PortfolioAliasRecord.portfolio_symbol, PortfolioAliasRecord.alias, unique=True)
Index("ix_portfolio_subsidiaries_symbol_name", PortfolioSubsidiaryRecord.portfolio_symbol, PortfolioSubsidiaryRecord.subsidiary_name, unique=True)


class AlertRecord(Base):
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    ticker: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    embedding_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    publication_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    processing_latency: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class DeduplicatedEventRecord(Base):
    __tablename__ = "deduplicated_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    ticker: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    embedding_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    publication_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    processing_latency: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_duplicate: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    duplicate_of: Mapped[str | None] = mapped_column(String(36), ForeignKey("news_events.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class LatencyMetricRecord(Base):
    __tablename__ = "latency_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    event_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    stage: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    ticker: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    embedding_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    publication_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    processing_latency: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


# Optimized indexes for new tables
Index("ix_news_events_ticker_pub_time", NewsEventRecord.ticker, NewsEventRecord.publication_time)
Index("ix_alerts_ticker_publication_time", AlertRecord.ticker, AlertRecord.publication_time)
Index("ix_deduplicated_events_ticker_pub_time", DeduplicatedEventRecord.ticker, DeduplicatedEventRecord.publication_time)
Index("ix_latency_metrics_stage_latency", LatencyMetricRecord.stage, LatencyMetricRecord.processing_latency)
