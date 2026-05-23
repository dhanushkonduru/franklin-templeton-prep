from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.services.deduplication import DeduplicationService
from app.services.embeddings import EmbeddingService, RedisEmbeddingCache
from app.normalization import normalize_event
from app.types import NormalizedNewsEvent


class FakeStore:
    def __init__(self, candidates: list[tuple[str, list[float]]] | None = None) -> None:
        self.candidates = candidates or []
        self.saved: list[tuple[str, list[float]]] = []

    async def save(self, event_id: str, embedding: list[float], *, ttl_seconds: int) -> None:
        self.saved.append((event_id, embedding))

    async def find_candidates(self, *, window_seconds: int, max_items: int) -> list[tuple[str, list[float]]]:
        return self.candidates


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def get(self, key: str):
        return self.values.get(key)

    async def set(self, key: str, value: str, *, ex: int | None = None):
        self.values[key] = value
        return True


@dataclass
class FakeEmbeddingModel:
    vectors: dict[str, list[float]]

    def encode(self, texts, normalize_embeddings=True):
        return [self.vectors[texts[0]]]


class FakeEmbeddingService(EmbeddingService):
    def __init__(self, cache=None):
        super().__init__(cache=cache)
        self._model = FakeEmbeddingModel(
            {
                "apple acquires ai startup": [1.0, 0.0],
                "apple buys artificial intelligence company": [0.99, 0.01],
            }
        )

    async def _load_model(self):
        return self._model


def test_deduplication_detects_similar_embedding():
    service = DeduplicationService(FakeStore([("previous", [1.0, 0.0])]), threshold=0.9)
    event = NormalizedNewsEvent(
        source="newsapi",
        source_event_id="current",
        title="Title",
        body="Body",
        url="https://example.com/current",
        published_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        tickers=(),
        company_name=None,
        content_hash="xyz",
        raw_payload={},
    )

    duplicate = __import__("asyncio").run(service.find_duplicate(event, [1.0, 0.0]))
    assert duplicate.duplicate_of == "previous"
    assert duplicate.similarity == 1.0


@pytest.mark.asyncio
async def test_embedding_cache_reuses_cached_vector():
    redis = FakeRedis()
    cache = RedisEmbeddingCache(redis)
    service = FakeEmbeddingService(cache=cache)

    first = await service.embed_text("apple acquires ai startup")
    second = await service.embed_text("apple acquires ai startup")

    assert first.embedding == [1.0, 0.0]
    assert second.embedding == [1.0, 0.0]
    assert len(redis.values) == 1


@pytest.mark.asyncio
async def test_semantic_duplicate_example_matches_on_similarity():
    redis = FakeRedis()
    cache = RedisEmbeddingCache(redis)
    service = FakeEmbeddingService(cache=cache)
    deduper = DeduplicationService(FakeStore([("event-1", [1.0, 0.0])]), threshold=0.95)

    event = NormalizedNewsEvent(
        source="newsapi",
        source_event_id="event-2",
        title="Apple buys artificial intelligence company",
        body="",
        url="https://example.com/apple-buys-ai-company",
        published_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        tickers=("AAPL",),
        company_name="Apple Inc.",
        content_hash="hash-2",
        raw_payload={},
    )

    embedding_result = await service.embed_text("apple buys artificial intelligence company")
    duplicate = await deduper.find_duplicate(event, embedding_result.embedding)

    assert duplicate.duplicate_of == "event-1"
    assert duplicate.similarity >= 0.95
