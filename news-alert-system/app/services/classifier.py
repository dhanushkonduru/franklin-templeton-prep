from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.services.embeddings import EmbeddingService, get_embedding_service


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class LabelDefinition:
    name: str
    keywords: tuple[str, ...]
    prototype_texts: tuple[str, ...]
    description: str = ""


@dataclass(slots=True)
class ClassificationResult:
    event_type: str
    confidence: float
    stage: str
    keyword_score: float = 0.0
    embedding_score: float = 0.0
    matched_keywords: tuple[str, ...] = field(default_factory=tuple)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "confidence": round(float(self.confidence), 4),
        }


DEFAULT_LABELS: tuple[LabelDefinition, ...] = (
    LabelDefinition(
        name="earnings",
        keywords=(
            "earnings",
            "revenue",
            "profit",
            "quarterly results",
            "guidance",
            "eps",
            "beats estimates",
            "beat estimates",
            "raises guidance",
            "raises outlook",
            "misses estimates",
            "full year outlook",
        ),
        prototype_texts=(
            "company reports quarterly earnings and revenue growth",
            "strong profit beats estimates and raises guidance",
            "quarterly results with improved outlook and margins",
        ),
        description="Financial results, guidance, and earnings announcements.",
    ),
    LabelDefinition(
        name="merger_acquisition",
        keywords=("acquires", "acquisition", "buys", "buying", "merger", "takeover", "deal", "purchase", "combination"),
        prototype_texts=(
            "company acquires rival in strategic deal",
            "merger agreement announced with target company",
            "takeover transaction and acquisition news",
        ),
        description="Mergers, acquisitions, takeovers, and strategic combinations.",
    ),
    LabelDefinition(
        name="leadership_change",
        keywords=(
            "ceo",
            "cfo",
            "cto",
            "chief executive",
            "appoints",
            "appointed",
            "resigns",
            "steps down",
            "joins",
            "departure",
            "transition",
            "leadership",
        ),
        prototype_texts=(
            "chief executive resigns and board names successor",
            "company appoints new chief financial officer",
            "leadership transition after executive departure",
        ),
        description="Executive appointments, resignations, and leadership transitions.",
    ),
    LabelDefinition(
        name="regulation",
        keywords=("sec", "regulator", "regulation", "regulatory", "compliance", "rule", "investigation", "fine", "probe", "antitrust", "approval"),
        prototype_texts=(
            "regulatory investigation and sec review",
            "new rule or compliance action from regulators",
            "antitrust or regulatory approval for company",
        ),
        description="Regulatory actions, rules, compliance, and investigations.",
    ),
    LabelDefinition(
        name="analyst_rating",
        keywords=("upgrade", "downgrade", "initiates coverage", "coverage", "price target", "buy rating", "sell rating", "overweight", "underweight", "analyst"),
        prototype_texts=(
            "analyst upgrades stock and raises price target",
            "coverage initiated with buy rating and price objective",
            "downgrade from analysts after valuation concerns",
        ),
        description="Analyst actions, rating changes, and price target revisions.",
    ),
    LabelDefinition(
        name="product_launch",
        keywords=("launches", "launch", "unveils", "introduces", "debut", "rolls out", "product", "service", "release"),
        prototype_texts=(
            "company launches new product and service",
            "product debut and platform rollout announcement",
            "unveils new device or software release",
        ),
        description="Product, service, and platform launches.",
    ),
    LabelDefinition(
        name="lawsuit",
        keywords=("lawsuit", "sues", "sued", "litigation", "complaint", "settlement", "legal action", "court", "class action", "trial"),
        prototype_texts=(
            "company faces lawsuit and litigation",
            "court complaint and legal action filed",
            "settlement agreement after class action litigation",
        ),
        description="Legal disputes, litigation, settlements, and lawsuits.",
    ),
    LabelDefinition(
        name="bankruptcy",
        keywords=("bankruptcy", "chapter 11", "insolvency", "restructuring", "default", "liquidation", "chapter 7", "distressed", "creditors"),
        prototype_texts=(
            "company files for bankruptcy under chapter 11",
            "insolvency, debt restructuring, and creditor negotiations",
            "liquidation or default after financial distress",
        ),
        description="Bankruptcy, insolvency, default, and debt restructuring.",
    ),
    LabelDefinition(
        name="other",
        keywords=(),
        prototype_texts=(
            "general corporate news without a clear catalyst",
            "miscellaneous financial news item",
            "other market-related headline",
        ),
        description="Fallback label when no specific class fits.",
    ),
)


def _normalize_headline(headline: str) -> str:
    return re.sub(r"\s+", " ", headline.lower()).strip()


def _confidence_floor(label: str) -> float:
    if label == "other":
        return 0.0
    return max(settings.classifier_min_confidence, 0.35)


class EventClassifier:
    def __init__(
        self,
        *,
        embedding_service: EmbeddingService | None = None,
        labels: Iterable[LabelDefinition] | None = None,
        keyword_threshold: float | None = None,
        embedding_threshold: float | None = None,
        other_max_confidence: float | None = None,
    ) -> None:
        self.embedding_service = embedding_service or get_embedding_service()
        self.keyword_threshold = keyword_threshold or settings.classifier_keyword_threshold
        self.embedding_threshold = embedding_threshold or settings.classifier_embedding_threshold
        self.other_max_confidence = other_max_confidence or settings.classifier_other_max_confidence
        self.labels = tuple(labels or DEFAULT_LABELS)
        self._prototype_embeddings: dict[str, list[list[float]]] = {}

    async def _ensure_prototype_embeddings(self) -> None:
        if self._prototype_embeddings:
            return

        for label in self.labels:
            embeddings: list[list[float]] = []
            if label.prototype_texts:
                results = await self.embedding_service.embed_events([text_to_event(text) for text in label.prototype_texts])
                embeddings = [result.embedding for result in results]
            self._prototype_embeddings[label.name] = embeddings

    def _keyword_match(self, headline: str) -> tuple[str, float, tuple[str, ...]]:
        normalized = _normalize_headline(headline)
        best_label = "other"
        best_score = 0.0
        best_matches: tuple[str, ...] = tuple()

        for label in self.labels:
            if not label.keywords:
                continue

            matches = tuple(sorted({keyword for keyword in label.keywords if keyword in normalized}))
            if not matches:
                continue

            multiword_bonus = sum(1 for keyword in matches if len(keyword.split()) > 1) * 0.05
            keyword_score = min(1.0, 0.40 + (0.22 * len(matches)) + multiword_bonus)
            if keyword_score > best_score:
                best_label = label.name
                best_score = keyword_score
                best_matches = matches

        return best_label, best_score, best_matches

    async def _embedding_match(self, headline: str) -> tuple[str, float]:
        await self._ensure_prototype_embeddings()
        headline_embedding = (await self.embedding_service.embed_text(headline)).embedding
        headline_array = np.asarray(headline_embedding, dtype=float).reshape(1, -1)

        best_label = "other"
        best_similarity = 0.0
        for label in self.labels:
            embeddings = self._prototype_embeddings.get(label.name, [])
            if not embeddings:
                continue
            prototype_array = np.asarray(embeddings, dtype=float)
            similarities = cosine_similarity(headline_array, prototype_array)[0]
            label_similarity = float(np.max(similarities))
            if label_similarity > best_similarity:
                best_similarity = label_similarity
                best_label = label.name

        return best_label, best_similarity

    def _score(self, *, keyword_score: float, embedding_score: float, chosen_label: str, stage: str) -> float:
        if chosen_label == "other":
            return min(self.other_max_confidence, max(keyword_score, embedding_score) * 0.6)

        if stage == "keyword":
            return min(0.99, max(_confidence_floor(chosen_label), 0.72 * keyword_score + 0.28 * embedding_score))

        return min(0.99, max(_confidence_floor(chosen_label), 0.55 * embedding_score + 0.25 * keyword_score + 0.20))

    async def classify_result(self, headline: str) -> ClassificationResult:
        keyword_label, keyword_score, matched_keywords = self._keyword_match(headline)
        embedding_label, embedding_score = await self._embedding_match(headline)

        if keyword_label != "other" and keyword_score >= self.keyword_threshold:
            confidence = self._score(keyword_score=keyword_score, embedding_score=embedding_score, chosen_label=keyword_label, stage="keyword")
            logger.info(
                "classified headline with keyword stage",
                extra={"headline": headline, "event_type": keyword_label, "confidence": round(confidence, 4), "stage": "keyword"},
            )
            return ClassificationResult(
                event_type=keyword_label,
                confidence=confidence,
                stage="keyword",
                keyword_score=keyword_score,
                embedding_score=embedding_score,
                matched_keywords=matched_keywords,
                rationale=f"keyword match: {', '.join(matched_keywords)}",
            )

        chosen_label = embedding_label if embedding_score >= self.embedding_threshold else "other"
        stage = "embedding" if chosen_label != "other" else "fallback"
        confidence = self._score(keyword_score=keyword_score, embedding_score=embedding_score, chosen_label=chosen_label, stage=stage)

        logger.info(
            "classified headline with embedding stage",
            extra={"headline": headline, "event_type": chosen_label, "confidence": round(confidence, 4), "stage": stage},
        )

        return ClassificationResult(
            event_type=chosen_label,
            confidence=confidence,
            stage=stage,
            keyword_score=keyword_score,
            embedding_score=embedding_score,
            matched_keywords=matched_keywords,
            rationale="embedding fallback" if chosen_label != "other" else "no strong class match",
        )

    async def classify(self, headline: str) -> dict[str, Any]:
        return (await self.classify_result(headline)).to_dict()


def text_to_event(headline: str):
    from datetime import datetime, timezone

    from app.types import NormalizedNewsEvent

    return NormalizedNewsEvent(
        source="classifier",
        source_event_id=headline,
        title=headline,
        body="",
        url="",
        published_at=datetime.now(timezone.utc),
        tickers=tuple(),
        company_name=None,
        content_hash=headline,
        raw_payload={},
    )


def get_event_classifier() -> EventClassifier:
    return EventClassifier()
