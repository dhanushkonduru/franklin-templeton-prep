from __future__ import annotations

from app.models import PortfolioHolding, PortfolioWatchlist
from app.services.matching import PortfolioMatcher
from app.types import NormalizedNewsEvent


def test_portfolio_matcher_matches_on_ticker_and_company_name():
    event = NormalizedNewsEvent(
        source="alpaca",
        source_event_id="event-2",
        title="Apple reports record quarter",
        body="Apple announced strong earnings.",
        url="https://example.com/apple",
        published_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        tickers=("AAPL",),
        company_name="Apple Inc.",
        content_hash="abc",
        raw_payload={},
    )
    holdings = [
        PortfolioHolding(portfolio_name="Core", ticker="AAPL", company_name="Apple Inc.", active=True),
        PortfolioHolding(portfolio_name="Energy", ticker="XOM", company_name="Exxon Mobil Corporation", active=True),
    ]

    matches = PortfolioMatcher().match(event, holdings, [PortfolioWatchlist(portfolio_name="Tech", keywords=["apple", "ai"], active=True)])

    assert [match.portfolio_name for match in matches] == ["Core", "Tech"]
