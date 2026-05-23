from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from functools import lru_cache
import json
from typing import Any

import numpy as np
from redis.asyncio import Redis
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from app.config import settings
from app.normalization import event_text
from app.types import NormalizedNewsEvent


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddingResult:
    embedding: list[float]
    model_name: str


class RedisEmbeddingCache:
    def __init__(self, redis: Redis | None, *, prefix: str = "news:embedding-cache") -> None:
        self.redis = redis
        self.prefix = prefix

    def _key(self, text: str, model_name: str) -> str:
        digest = hashlib.sha256()
        digest.update(model_name.encode("utf-8"))
        digest.update(b"|")
        digest.update(text.strip().lower().encode("utf-8"))
        return f"{self.prefix}:{digest.hexdigest()}"

    async def get(self, text: str, model_name: str) -> list[float] | None:
        if self.redis is None:
            return None
        payload = await self.redis.get(self._key(text, model_name))
        if not payload:
            return None
        decoded = json.loads(payload)
        return [float(value) for value in decoded]

    async def set(self, text: str, model_name: str, embedding: list[float], ttl_seconds: int) -> None:
        if self.redis is None:
            return
        await self.redis.set(self._key(text, model_name), json.dumps(embedding), ex=ttl_seconds)


class EmbeddingService:
    def __init__(self, model_name: str | None = None, *, cache: RedisEmbeddingCache | None = None) -> None:
        self.model_name = model_name or settings.embedding_model_name
        self._model: object | None = None
        self._cache = cache or RedisEmbeddingCache(None)
        self._hashing_vectorizer = HashingVectorizer(n_features=384, alternate_sign=False, norm=None)

    @property
    def model(self) -> object | None:
        return self._model

    async def _load_model(self) -> object | None:
        if self._model is not None:
            return self._model

        def _load() -> object | None:
            try:
                from sentence_transformers import SentenceTransformer

                return SentenceTransformer(self.model_name)
            except Exception as exc:  # pragma: no cover - fallback depends on environment
                logger.warning("falling back to hashing embeddings because %s could not load: %s", self.model_name, exc)
                return None

        self._model = await asyncio.to_thread(_load)
        return self._model

    async def embed_text(self, text: str) -> EmbeddingResult:
        cached_embedding = await self._cache.get(text, self.model_name)
        if cached_embedding is not None:
            return EmbeddingResult(embedding=cached_embedding, model_name=self.model_name)

        model = await self._load_model()
        if model is not None:
            vector = await asyncio.to_thread(model.encode, [text], normalize_embeddings=True)
            embedding = [float(value) for value in np.asarray(vector[0], dtype=float)]
            await self._cache.set(text, self.model_name, embedding, settings.embedding_cache_ttl_seconds)
            return EmbeddingResult(embedding=embedding, model_name=self.model_name)

        hashed = self._hashing_vectorizer.transform([text])
        normalized = normalize(hashed, norm="l2")
        dense = normalized.toarray()[0].astype(float).tolist()
        await self._cache.set(text, self.model_name, dense, settings.embedding_cache_ttl_seconds)
        return EmbeddingResult(embedding=dense, model_name="hashing-vectorizer")

    async def embed_event(self, event: NormalizedNewsEvent) -> EmbeddingResult:
        return await self.embed_text(event_text(event))

    async def embed_events(self, events: list[NormalizedNewsEvent]) -> list[EmbeddingResult]:
        return await asyncio.gather(*(self.embed_event(event) for event in events))


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()

