from __future__ import annotations

from dataclasses import dataclass

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import PortfolioHolding, PortfolioWatchlist
from app.normalization import event_text
from app.repositories import AlertRepository
from app.schemas import PortfolioMatchRequest
from app.services.portfolio_matching import PortfolioMatchingService
from app.types import NormalizedNewsEvent
from app.utils.text import normalize_text, normalize_ticker


@dataclass(slots=True)
class MatchResult:
    portfolio_name: str
    score: float
    holding: PortfolioHolding | None = None
    matched_symbol: str | None = None
    match_reason: str = ""


class PortfolioMatcher:
    def match(
        self,
        event: NormalizedNewsEvent,
        holdings: list[PortfolioHolding],
        watchlists: list[PortfolioWatchlist] | None = None,
    ) -> list[MatchResult]:
        normalized_text = normalize_text(event_text(event))
        ticker_set = {ticker.upper() for ticker in event.tickers}
        matches: dict[str, MatchResult] = {}

        for holding in holdings:
            if not holding.active:
                continue

            score = 0.0
            reason = ""
            if holding.ticker.upper() in ticker_set:
                score += 1.0
                reason = "ticker"
            if holding.ticker.lower() in normalized_text:
                score += 0.45
                reason = reason or "ticker_text"
            if normalize_text(holding.company_name) in normalized_text or normalized_text in normalize_text(holding.company_name):
                score += 0.65
                reason = reason or "company_name"

            if score >= 0.5:
                previous = matches.get(holding.portfolio_name)
                if previous is None or score > previous.score:
                    matches[holding.portfolio_name] = MatchResult(
                        portfolio_name=holding.portfolio_name,
                        score=score,
                        holding=holding,
                        matched_symbol=holding.ticker,
                        match_reason=reason,
                    )

        if watchlists:
            for watchlist in watchlists:
                if not watchlist.active:
                    continue
                keyword_score = sum(0.2 for keyword in watchlist.keywords if normalize_text(keyword) in normalized_text)
                if keyword_score > 0.0:
                    previous = matches.get(watchlist.portfolio_name)
                    score = min(1.0, keyword_score)
                    if previous is None or score > previous.score:
                        matches[watchlist.portfolio_name] = MatchResult(
                            portfolio_name=watchlist.portfolio_name,
                            score=score,
                            match_reason="watchlist_keyword",
                        )

        return sorted(matches.values(), key=lambda item: item.score, reverse=True)


class UnifiedPortfolioMatcher:
    """Combines alias/subsidiary matching with holdings and watchlist keyword rules."""

    def __init__(self, *, redis: Redis | None = None) -> None:
        self.redis = redis
        self.legacy_matcher = PortfolioMatcher()

    async def match(
        self,
        event: NormalizedNewsEvent,
        event_type: str,
        *,
        session: AsyncSession,
    ) -> list[MatchResult]:
        repository = AlertRepository(session)
        holdings = await repository.list_holdings(active_only=True)
        watchlists = await repository.list_watchlists(active_only=True)

        primary_ticker = list(event.tickers)[0] if event.tickers else ""
        portfolio_service = PortfolioMatchingService(redis=self.redis, session=session)
        match_response = await portfolio_service.match(
            PortfolioMatchRequest(
                headline=event.title,
                ticker=primary_ticker,
                event_type=event_type,
            )
        )

        matches: dict[str, MatchResult] = {}
        if match_response.portfolio_hit and match_response.matched_symbol:
            symbol = normalize_ticker(match_response.matched_symbol)
            for holding in holdings:
                if normalize_ticker(holding.ticker) == symbol:
                    matches[holding.portfolio_name] = MatchResult(
                        portfolio_name=holding.portfolio_name,
                        score=1.0,
                        holding=holding,
                        matched_symbol=symbol,
                        match_reason="portfolio_universe",
                    )

        for legacy_match in self.legacy_matcher.match(event, holdings, watchlists):
            previous = matches.get(legacy_match.portfolio_name)
            if previous is None or legacy_match.score > previous.score:
                matches[legacy_match.portfolio_name] = legacy_match

        return sorted(matches.values(), key=lambda item: item.score, reverse=True)
