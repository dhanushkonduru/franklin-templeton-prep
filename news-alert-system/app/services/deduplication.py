from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import numpy as np
from redis.asyncio import Redis
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.streaming import embedding_from_json, embedding_to_json
from app.types import NormalizedNewsEvent


logger = logging.getLogger(__name__)


class DeduplicationStore(Protocol):
    async def save(self, event_id: str, embedding: list[float], *, ttl_seconds: int) -> None: ...

    async def find_candidates(self, *, window_seconds: int, max_items: int) -> list[tuple[str, list[float]]]: ...


@dataclass(slots=True)
class DuplicateMatch:
    duplicate_of: str | None
    similarity: float


class RedisDeduplicationStore:
    def __init__(self, redis: Redis, namespace: str = "dedup") -> None:
        self.redis = redis
        self.namespace = namespace

    def _index_key(self) -> str:
        return f"{self.namespace}:index"

    def _event_key(self, event_id: str) -> str:
        return f"{self.namespace}:event:{event_id}"

    async def save(self, event_id: str, embedding: list[float], *, ttl_seconds: int) -> None:
        now = datetime.now(timezone.utc).timestamp()
        payload = json.dumps({"event_id": event_id, "embedding": embedding_to_json(embedding), "timestamp": now})
        pipe = self.redis.pipeline()
        pipe.set(self._event_key(event_id), payload, ex=ttl_seconds)
        pipe.zadd(self._index_key(), {event_id: now})
        pipe.zremrangebyscore(self._index_key(), 0, now - ttl_seconds)
        await pipe.execute()

    async def find_candidates(self, *, window_seconds: int, max_items: int) -> list[tuple[str, list[float]]]:
        now = datetime.now(timezone.utc).timestamp()
        minimum_score = now - window_seconds
        event_ids = await self.redis.zrevrangebyscore(self._index_key(), now, minimum_score, start=0, num=max_items)
        if not event_ids:
            return []

        keys = [self._event_key(event_id) for event_id in event_ids]
        payloads = await self.redis.mget(keys)
        candidates: list[tuple[str, list[float]]] = []
        for payload in payloads:
            if not payload:
                continue
            decoded = json.loads(payload)
            candidates.append((decoded["event_id"], embedding_from_json(decoded["embedding"])))
        return candidates


class DeduplicationService:
    def __init__(self, store: DeduplicationStore, threshold: float | None = None) -> None:
        self.store = store
        self.threshold = threshold or settings.dedup_similarity_threshold

    async def find_duplicate(self, event: NormalizedNewsEvent, embedding: list[float]) -> DuplicateMatch:
        candidates = await self.store.find_candidates(
            window_seconds=settings.redis_recent_window_seconds,
            max_items=settings.dedup_neighborhood_size,
        )
        if not candidates:
            return DuplicateMatch(duplicate_of=None, similarity=0.0)

        best_duplicate: str | None = None
        best_similarity = 0.0
        embedding_array = np.asarray(embedding, dtype=float).reshape(1, -1)

        for event_id, candidate_embedding in candidates:
            candidate_array = np.asarray(candidate_embedding, dtype=float).reshape(1, -1)
            if embedding_array.shape[1] != candidate_array.shape[1]:
                continue
            similarity = float(cosine_similarity(embedding_array, candidate_array)[0][0])
            if similarity > best_similarity:
                best_similarity = similarity
                best_duplicate = event_id

        if best_duplicate and best_similarity >= self.threshold:
            logger.info("duplicate detected for %s against %s with similarity %.4f", event.source_event_id, best_duplicate, best_similarity)
            return DuplicateMatch(duplicate_of=best_duplicate, similarity=best_similarity)

        return DuplicateMatch(duplicate_of=None, similarity=best_similarity)

    async def remember(self, event: NormalizedNewsEvent, embedding: list[float]) -> None:
        await self.store.save(event.source_event_id, embedding, ttl_seconds=settings.dedup_ttl_seconds)
