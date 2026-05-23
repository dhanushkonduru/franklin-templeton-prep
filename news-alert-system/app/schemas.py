from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    time: datetime


class PortfolioHoldingCreate(BaseModel):
    portfolio_name: str = Field(min_length=1, max_length=128)
    ticker: str = Field(min_length=1, max_length=16)
    company_name: str = Field(min_length=1, max_length=255)
    active: bool = True
    webhook_url: str | None = None
    slack_webhook_url: str | None = None
    email_address: str | None = None


class PortfolioAliasCreate(BaseModel):
    portfolio_symbol: str = Field(min_length=1, max_length=16)
    alias: str = Field(min_length=1, max_length=255)
    active: bool = True


class PortfolioSubsidiaryCreate(BaseModel):
    portfolio_symbol: str = Field(min_length=1, max_length=16)
    subsidiary_name: str = Field(min_length=1, max_length=255)
    active: bool = True


class PortfolioMatchRequest(BaseModel):
    headline: str = Field(min_length=1, max_length=500)
    ticker: str = Field(default="", max_length=16)
    event_type: str = Field(min_length=1, max_length=64)
    portfolio_symbols: list[str] = Field(default_factory=list)


class PortfolioMatchResponse(BaseModel):
    portfolio_hit: bool
    matched_symbol: str | None = None


class PortfolioAliasRead(PortfolioAliasCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime


class PortfolioSubsidiaryRead(PortfolioSubsidiaryCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime


class PortfolioHoldingRead(PortfolioHoldingCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime


class NewsEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source: str
    source_event_id: str
    title: str
    body: str
    url: str
    published_at: datetime
    ticker_symbols: list[str]
    company_name: str | None
    content_hash: str
    event_type: str
    event_confidence: float
    similarity_score: float
    duplicate_of: str | None
    created_at: datetime


class IngestEventRequest(BaseModel):
    source: str = Field(min_length=1, max_length=64)
    source_event_id: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1)
    body: str = ""
    url: str = Field(min_length=1)
    published_at: datetime
    tickers: list[str] = Field(default_factory=list)
    company_name: str | None = None
    raw_payload: dict = Field(default_factory=dict)


class IngestedEventResponse(BaseModel):
    accepted: bool
    source: str
    source_event_id: str
    stream_id: str | None = None
