from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.pipeline.orchestrator import NewsPipeline


@pytest.mark.asyncio
async def test_pipeline_skips_delivery_for_duplicates(sample_raw_event):
    redis = MagicMock()
    session_factory = MagicMock()

    pipeline = NewsPipeline(redis=redis, session_factory=session_factory)
    pipeline.embedding_service = AsyncMock()
    pipeline.embedding_service.embed_event.return_value = MagicMock(embedding=[0.1, 0.2, 0.3])
    pipeline.classifier = AsyncMock()
    pipeline.classifier.classify.return_value = {"event_type": "earnings", "confidence": 0.9}
    pipeline.dedup_service = AsyncMock()
    pipeline.dedup_service.find_duplicate.return_value = MagicMock(duplicate_of="prior-event", similarity=0.95)
    pipeline.dedup_service.remember = AsyncMock()
    pipeline.matcher = AsyncMock()
    pipeline.matcher.match.return_value = [MagicMock(portfolio_name="Core", holding=MagicMock(), score=1.0)]
    pipeline.notification_service = AsyncMock()

    mock_session = AsyncMock()
    session_factory.return_value.__aenter__.return_value = mock_session
    session_factory.return_value.__aexit__.return_value = None

    mock_repository = AsyncMock()
    mock_repository.find_event_id_by_source_event_id.return_value = "db-event-1"
    mock_repository.upsert_event.return_value = MagicMock(id="event-123")
    mock_repository.create_deduplicated_event = AsyncMock()
    mock_repository.record_latency_metric = AsyncMock()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("app.pipeline.orchestrator.AlertRepository", lambda _session: mock_repository)
    result = await pipeline.process_raw_event(sample_raw_event)
    monkeypatch.undo()

    assert result.is_duplicate is True
    assert result.notifications_sent == 0
    pipeline.notification_service.dispatch.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_dispatches_when_matched_and_unique(sample_raw_event):
    redis = MagicMock()
    session_factory = MagicMock()

    pipeline = NewsPipeline(redis=redis, session_factory=session_factory)
    pipeline.embedding_service = AsyncMock()
    pipeline.embedding_service.embed_event.return_value = MagicMock(embedding=[0.1, 0.2, 0.3])
    pipeline.classifier = AsyncMock()
    pipeline.classifier.classify.return_value = {"event_type": "earnings", "confidence": 0.9}
    pipeline.dedup_service = AsyncMock()
    pipeline.dedup_service.find_duplicate.return_value = MagicMock(duplicate_of=None, similarity=0.1)
    pipeline.dedup_service.remember = AsyncMock()

    holding = MagicMock()
    pipeline.matcher = AsyncMock()
    pipeline.matcher.match.return_value = [MagicMock(portfolio_name="Core", holding=holding, score=1.0)]
    pipeline.notification_service = AsyncMock()

    mock_session = AsyncMock()
    session_factory.return_value.__aenter__.return_value = mock_session
    session_factory.return_value.__aexit__.return_value = None

    mock_repository = AsyncMock()
    mock_repository.find_event_id_by_source_event_id.return_value = None
    mock_repository.upsert_event.return_value = MagicMock(id="event-123")
    mock_repository.create_deduplicated_event = AsyncMock()
    mock_repository.create_alert = AsyncMock()
    mock_repository.record_latency_metric = AsyncMock()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("app.pipeline.orchestrator.AlertRepository", lambda _session: mock_repository)
    result = await pipeline.process_raw_event(sample_raw_event)
    monkeypatch.undo()

    assert result.is_duplicate is False
    assert result.notifications_sent == 1
    pipeline.notification_service.dispatch.assert_awaited_once()
