from __future__ import annotations

import logging
from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import settings
from app.models import PortfolioMatchHistory
from app.redis_client import get_json, set_json
from app.repositories import AlertRepository
from app.schemas import PortfolioMatchRequest, PortfolioMatchResponse
from app.utils.text import normalize_text, normalize_ticker


logger = logging.getLogger(__name__)


DEFAULT_ALIAS_MAP: dict[str, str] = {
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "facebook": "META",
    "meta": "META",
    "instagram": "META",
    "whatsapp": "META",
    "amazon": "AMZN",
    "aws": "AMZN",
    "apple": "AAPL",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "microsoft": "MSFT",
}

DEFAULT_SUBSIDIARY_MAP: dict[str, str] = {
    "waymo": "GOOGL",
    "instagram": "META",
    "whatsapp": "META",
    "aws": "AMZN",
    "x.ai": "TSLA",
}


@dataclass(slots=True)
class PortfolioUniverse:
    symbols: tuple[str, ...]
    aliases: dict[str, str]
    subsidiaries: dict[str, str]

    def fingerprint(self) -> str:
        digest = sha256()
        digest.update("|".join(sorted(self.symbols)).encode("utf-8"))
        digest.update(b"|")
        digest.update("|".join(f"{alias}:{symbol}" for alias, symbol in sorted(self.aliases.items())).encode("utf-8"))
        digest.update(b"|")
        digest.update("|".join(f"{name}:{symbol}" for name, symbol in sorted(self.subsidiaries.items())).encode("utf-8"))
        return digest.hexdigest()


class PortfolioMatchingService:
    def __init__(
        self,
        *,
        redis: Redis | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        session: AsyncSession | None = None,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        self.redis = redis
        self.session_factory = session_factory
        self.session = session
        self.cache_ttl_seconds = cache_ttl_seconds or settings.portfolio_match_cache_ttl_seconds

    def _normalize_symbols(self, symbols: Sequence[str]) -> tuple[str, ...]:
        return tuple(sorted({normalize_ticker(symbol) for symbol in symbols if str(symbol).strip()}))

    def _normalize_alias_map(self, rows: dict[str, str]) -> dict[str, str]:
        return {normalize_text(alias): normalize_ticker(symbol) for alias, symbol in rows.items() if alias and symbol}

    def _normalize_subsidiary_map(self, rows: dict[str, str]) -> dict[str, str]:
        return {normalize_text(name): normalize_ticker(symbol) for name, symbol in rows.items() if name and symbol}

    def _response_cache_key(self, request: PortfolioMatchRequest, universe: PortfolioUniverse) -> str:
        digest = sha256()
        digest.update(normalize_text(request.headline).encode("utf-8"))
        digest.update(b"|")
        digest.update(normalize_ticker(request.ticker).encode("utf-8"))
        digest.update(b"|")
        digest.update(normalize_text(request.event_type).encode("utf-8"))
        digest.update(b"|")
        digest.update(universe.fingerprint().encode("utf-8"))
        return f"portfolio:match:{digest.hexdigest()}"

    async def _load_universe_from_db(self) -> PortfolioUniverse:
        if self.session is None and self.session_factory is None:
            return PortfolioUniverse(symbols=tuple(), aliases=DEFAULT_ALIAS_MAP.copy(), subsidiaries=DEFAULT_SUBSIDIARY_MAP.copy())

        if self.session is not None:
            repository = AlertRepository(self.session)
            holdings = await repository.list_holdings(active_only=True)
            aliases = await repository.list_aliases(active_only=True)
            subsidiaries = await repository.list_subsidiaries(active_only=True)
        else:
            async with self.session_factory() as session:
                repository = AlertRepository(session)
                holdings = await repository.list_holdings(active_only=True)
                aliases = await repository.list_aliases(active_only=True)
                subsidiaries = await repository.list_subsidiaries(active_only=True)

        symbol_set = self._normalize_symbols([holding.ticker for holding in holdings])
        alias_map = DEFAULT_ALIAS_MAP.copy()
        alias_map.update({alias.alias: alias.portfolio_symbol for alias in aliases})
        subsidiary_map = DEFAULT_SUBSIDIARY_MAP.copy()
        subsidiary_map.update({item.subsidiary_name: item.portfolio_symbol for item in subsidiaries})
        return PortfolioUniverse(
            symbols=symbol_set,
            aliases=self._normalize_alias_map(alias_map),
            subsidiaries=self._normalize_subsidiary_map(subsidiary_map),
        )

    async def _load_universe(self, portfolio_symbols: Sequence[str] | None = None) -> PortfolioUniverse:
        if portfolio_symbols:
            symbols = self._normalize_symbols(portfolio_symbols)
            return PortfolioUniverse(
                symbols=symbols,
                aliases=self._normalize_alias_map(DEFAULT_ALIAS_MAP),
                subsidiaries=self._normalize_subsidiary_map(DEFAULT_SUBSIDIARY_MAP),
            )

        cache_key = "portfolio:universe:v1"
        if self.redis is not None:
            cached = await get_json(self.redis, cache_key)
            if cached is not None:
                return PortfolioUniverse(
                    symbols=tuple(cached.get("symbols", [])),
                    aliases={str(key): str(value) for key, value in cached.get("aliases", {}).items()},
                    subsidiaries={str(key): str(value) for key, value in cached.get("subsidiaries", {}).items()},
                )

        universe = await self._load_universe_from_db()
        if self.redis is not None:
            await set_json(
                self.redis,
                cache_key,
                {
                    "symbols": list(universe.symbols),
                    "aliases": universe.aliases,
                    "subsidiaries": universe.subsidiaries,
                },
                self.cache_ttl_seconds,
            )
        return universe

    def _match_exact_ticker(self, ticker: str, universe: PortfolioUniverse) -> str | None:
        normalized_ticker = normalize_ticker(ticker)
        if normalized_ticker and normalized_ticker in universe.symbols:
            return normalized_ticker
        return None

    def _match_alias(self, headline: str, universe: PortfolioUniverse) -> tuple[str | None, str | None]:
        normalized_headline = normalize_text(headline)
        for alias, symbol in sorted(universe.aliases.items(), key=lambda item: len(item[0]), reverse=True):
            if alias and alias in normalized_headline and symbol in universe.symbols:
                return symbol, alias
        return None, None

    def _match_subsidiary(self, headline: str, universe: PortfolioUniverse) -> tuple[str | None, str | None]:
        normalized_headline = normalize_text(headline)
        for subsidiary, symbol in sorted(universe.subsidiaries.items(), key=lambda item: len(item[0]), reverse=True):
            if subsidiary and subsidiary in normalized_headline and symbol in universe.symbols:
                return symbol, subsidiary
        return None, None

    async def match(self, request: PortfolioMatchRequest, portfolio_symbols: Sequence[str] | None = None) -> PortfolioMatchResponse:
        universe = await self._load_universe(portfolio_symbols or request.portfolio_symbols)
        cache_key = self._response_cache_key(request, universe)

        if self.redis is not None:
            cached = await get_json(self.redis, cache_key)
            if cached is not None:
                logger.info("portfolio match cache hit", extra={"headline": request.headline, "ticker": request.ticker, "event_type": request.event_type})
                response = PortfolioMatchResponse(**cached)
                if self.session is not None:
                    repository = AlertRepository(self.session)
                    await repository.record_portfolio_match(
                        PortfolioMatchHistory(
                            headline=request.headline,
                            ticker=normalize_ticker(request.ticker),
                            event_type=request.event_type,
                            portfolio_hit=response.portfolio_hit,
                            matched_symbol=response.matched_symbol,
                            match_reason="cache_hit",
                            source_payload={**request.model_dump(), "cache_hit": True},
                        )
                    )
                elif self.session_factory is not None:
                    async with self.session_factory() as session:
                        repository = AlertRepository(session)
                        await repository.record_portfolio_match(
                            PortfolioMatchHistory(
                                headline=request.headline,
                                ticker=normalize_ticker(request.ticker),
                                event_type=request.event_type,
                                portfolio_hit=response.portfolio_hit,
                                matched_symbol=response.matched_symbol,
                                match_reason="cache_hit",
                                source_payload={**request.model_dump(), "cache_hit": True},
                            )
                        )
                return response

        matched_symbol = self._match_exact_ticker(request.ticker, universe)
        match_reason = "ticker"

        if matched_symbol is None:
            matched_symbol, alias = self._match_alias(request.headline, universe)
            match_reason = f"alias:{alias}" if alias else ""

        if matched_symbol is None:
            matched_symbol, subsidiary = self._match_subsidiary(request.headline, universe)
            match_reason = f"subsidiary:{subsidiary}" if subsidiary else ""

        portfolio_hit = matched_symbol is not None
        response = PortfolioMatchResponse(portfolio_hit=portfolio_hit, matched_symbol=matched_symbol)

        if self.redis is not None:
            await set_json(self.redis, cache_key, response.model_dump(), self.cache_ttl_seconds)

        if self.session is not None:
            repository = AlertRepository(self.session)
            await repository.record_portfolio_match(
                PortfolioMatchHistory(
                    headline=request.headline,
                    ticker=normalize_ticker(request.ticker),
                    event_type=request.event_type,
                    portfolio_hit=portfolio_hit,
                    matched_symbol=matched_symbol,
                    match_reason=match_reason or "no_match",
                    source_payload=request.model_dump(),
                )
            )
        elif self.session_factory is not None:
            async with self.session_factory() as session:
                repository = AlertRepository(session)
                await repository.record_portfolio_match(
                    PortfolioMatchHistory(
                        headline=request.headline,
                        ticker=normalize_ticker(request.ticker),
                        event_type=request.event_type,
                        portfolio_hit=portfolio_hit,
                        matched_symbol=matched_symbol,
                        match_reason=match_reason or "no_match",
                        source_payload=request.model_dump(),
                    )
                )

        logger.info(
            "portfolio match evaluated",
            extra={
                "headline": request.headline,
                "ticker": request.ticker,
                "event_type": request.event_type,
                "portfolio_hit": portfolio_hit,
                "matched_symbol": matched_symbol,
                "reason": match_reason or "no_match",
            },
        )
        return response
