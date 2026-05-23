from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.normalization import normalize_event
from app.repositories import AlertRepository
from app.services.classifier import EventClassifier, get_event_classifier
from app.services.deduplication import DeduplicationService, DuplicateMatch, RedisDeduplicationStore
from app.services.delivery import NotificationService
from app.services.embeddings import EmbeddingService, RedisEmbeddingCache, get_embedding_service
from app.services.matching import UnifiedPortfolioMatcher
from app.types import NormalizedNewsEvent, RawNewsEvent
from app.utils.text import compute_embedding_hash


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineStageLatency:
    normalization: float = 0.0
    embedding: float = 0.0
    deduplication: float = 0.0
    classification: float = 0.0
    matching: float = 0.0
    delivery: float = 0.0
    total: float = 0.0


@dataclass(slots=True)
class PipelineResult:
    event: NormalizedNewsEvent
    event_type: str
    confidence: float
    duplicate_of: str | None
    duplicate_source_event_id: str | None
    similarity_score: float
    is_duplicate: bool
    matched_portfolios: list[str]
    persisted_event_id: str
    notifications_sent: int
    stage_latency: PipelineStageLatency


class NewsPipeline:
    """Coordinates the full processing path from normalized event to persistence and delivery."""

    def __init__(
        self,
        *,
        redis: Redis,
        session_factory: async_sessionmaker[AsyncSession],
        embedding_service: EmbeddingService | None = None,
        classifier: EventClassifier | None = None,
        dedup_service: DeduplicationService | None = None,
        matcher: UnifiedPortfolioMatcher | None = None,
        notification_service: NotificationService | None = None,
    ) -> None:
        self.redis = redis
        self.session_factory = session_factory
        self.embedding_service = embedding_service or EmbeddingService(cache=RedisEmbeddingCache(redis))
        self.classifier = classifier or get_event_classifier()
        self.dedup_service = dedup_service or DeduplicationService(RedisDeduplicationStore(redis))
        self.matcher = matcher or UnifiedPortfolioMatcher(redis=redis)
        self.notification_service = notification_service or NotificationService(redis=redis)

    async def _resolve_duplicate_event_id(
        self,
        repository: AlertRepository,
        duplicate_match: DuplicateMatch,
        event: NormalizedNewsEvent,
    ) -> str | None:
        if duplicate_match.duplicate_of is None:
            return None
        resolved = await repository.find_event_id_by_source_event_id(event.source, duplicate_match.duplicate_of)
        if resolved is not None:
            return resolved
        return duplicate_match.duplicate_of

    async def process_raw_event(self, raw_event: RawNewsEvent) -> PipelineResult:
        start_total = time.perf_counter()
        stage_latency = PipelineStageLatency()

        start_norm = time.perf_counter()
        normalized_event = normalize_event(raw_event)
        stage_latency.normalization = time.perf_counter() - start_norm

        start_embed = time.perf_counter()
        embedding_result = await self.embedding_service.embed_event(normalized_event)
        stage_latency.embedding = time.perf_counter() - start_embed
        emb_hash = compute_embedding_hash(embedding_result.embedding)

        start_dedup = time.perf_counter()
        duplicate_match = await self.dedup_service.find_duplicate(normalized_event, embedding_result.embedding)
        stage_latency.deduplication = time.perf_counter() - start_dedup
        is_duplicate = duplicate_match.duplicate_of is not None

        start_classify = time.perf_counter()
        classification = await self.classifier.classify(normalized_event.title)
        event_type = classification["event_type"]
        confidence = float(classification["confidence"])
        stage_latency.classification = time.perf_counter() - start_classify

        primary_ticker = list(normalized_event.tickers)[0] if normalized_event.tickers else ""
        matched: list = []
        notifications_sent = 0
        persisted_event_id = ""
        resolved_duplicate_id: str | None = None

        async with self.session_factory() as session:
            repository = AlertRepository(session)
            resolved_duplicate_id = await self._resolve_duplicate_event_id(repository, duplicate_match, normalized_event)

            start_match = time.perf_counter()
            matched = await self.matcher.match(
                normalized_event,
                event_type,
                session=session,
            )
            stage_latency.matching = time.perf_counter() - start_match

            persisted = await repository.upsert_event(
                normalized_event,
                event_type=event_type,
                confidence=confidence,
                similarity_score=duplicate_match.similarity,
                duplicate_of=resolved_duplicate_id,
                embedding=embedding_result.embedding,
                embedding_hash=emb_hash,
                processing_latency=time.perf_counter() - start_total,
            )
            persisted_event_id = persisted.id

            await repository.create_deduplicated_event(
                headline=normalized_event.title,
                ticker=primary_ticker,
                embedding_hash=emb_hash,
                event_type=event_type,
                confidence=confidence,
                publication_time=normalized_event.published_at,
                processing_latency=time.perf_counter() - start_total,
                similarity_score=duplicate_match.similarity,
                is_duplicate=is_duplicate,
                duplicate_of=resolved_duplicate_id,
            )

            if matched and not is_duplicate:
                start_delivery = time.perf_counter()
                await self.notification_service.dispatch(
                    session,
                    persisted.id,
                    normalized_event,
                    event_type,
                    [result.holding for result in matched if result.holding is not None],
                    duplicate_match.similarity,
                    confidence,
                )
                notifications_sent = len(matched)
                stage_latency.delivery = time.perf_counter() - start_delivery

                for match_res in matched:
                    await repository.create_alert(
                        headline=normalized_event.title,
                        ticker=match_res.portfolio_name,
                        embedding_hash=emb_hash,
                        event_type=event_type,
                        confidence=confidence,
                        publication_time=normalized_event.published_at,
                        processing_latency=time.perf_counter() - start_total,
                    )

            stage_latency.total = time.perf_counter() - start_total
            await self._record_stage_metrics(
                repository,
                event_id=persisted_event_id,
                normalized_event=normalized_event,
                primary_ticker=primary_ticker,
                emb_hash=emb_hash,
                event_type=event_type,
                confidence=confidence,
                stage_latency=stage_latency,
            )

        await self.dedup_service.remember(normalized_event, embedding_result.embedding)

        return PipelineResult(
            event=normalized_event,
            event_type=event_type,
            confidence=confidence,
            duplicate_of=resolved_duplicate_id,
            duplicate_source_event_id=duplicate_match.duplicate_of,
            similarity_score=duplicate_match.similarity,
            is_duplicate=is_duplicate,
            matched_portfolios=[result.portfolio_name for result in matched],
            persisted_event_id=persisted_event_id,
            notifications_sent=notifications_sent,
            stage_latency=stage_latency,
        )

    async def _record_stage_metrics(
        self,
        repository: AlertRepository,
        *,
        event_id: str,
        normalized_event: NormalizedNewsEvent,
        primary_ticker: str,
        emb_hash: str,
        event_type: str,
        confidence: float,
        stage_latency: PipelineStageLatency,
    ) -> None:
        common = {
            "event_id": event_id,
            "headline": normalized_event.title,
            "ticker": primary_ticker,
            "embedding_hash": emb_hash,
            "event_type": event_type,
            "confidence": confidence,
            "publication_time": normalized_event.published_at,
        }
        stages = [
            ("normalization", stage_latency.normalization),
            ("embedding", stage_latency.embedding),
            ("deduplication", stage_latency.deduplication),
            ("classification", stage_latency.classification),
            ("matching", stage_latency.matching),
            ("total", stage_latency.total),
        ]
        if stage_latency.delivery > 0.0:
            stages.insert(-1, ("delivery", stage_latency.delivery))

        for stage_name, latency in stages:
            await repository.record_latency_metric(stage=stage_name, processing_latency=latency, **common)
