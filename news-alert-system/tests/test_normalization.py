from __future__ import annotations

from app.normalization import normalize_event


def test_normalize_event_sorts_tickers_and_hashes_content(sample_raw_event):
    normalized = normalize_event(sample_raw_event)

    assert normalized.tickers == ("AAPL",)
    assert normalized.company_name == "Apple Inc."
    assert len(normalized.content_hash) == 64
    assert normalized.title == sample_raw_event.title
