from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.services.classifier import EventClassifier


@dataclass(slots=True)
class FakeEmbeddingResult:
    embedding: list[float]
    model_name: str = "fake"


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.vector_map = {
            "company reports quarterly earnings and revenue growth": [1.0, 0.0, 0.0],
            "strong profit beats estimates and raises guidance": [1.0, 0.0, 0.0],
            "quarterly results with improved outlook and margins": [1.0, 0.0, 0.0],
            "company acquires rival in strategic deal": [0.0, 1.0, 0.0],
            "merger agreement announced with target company": [0.0, 1.0, 0.0],
            "takeover transaction and acquisition news": [0.0, 1.0, 0.0],
            "chief executive resigns and board names successor": [0.0, 0.0, 1.0],
            "company appoints new chief financial officer": [0.0, 0.0, 1.0],
            "leadership transition after executive departure": [0.0, 0.0, 1.0],
            "top executive exits in management shakeup": [0.0, 0.0, 1.0],
            "analyst upgrades stock and raises price target": [0.5, 0.5, 0.0],
            "coverage initiated with buy rating and price objective": [0.5, 0.5, 0.0],
            "downgrade from analysts after valuation concerns": [0.5, 0.5, 0.0],
            "company launches new product and service": [0.0, 0.5, 0.5],
            "product debut and platform rollout announcement": [0.0, 0.5, 0.5],
            "unveils new device or software release": [0.0, 0.5, 0.5],
            "company faces lawsuit and litigation": [0.0, 0.0, 0.0],
            "court complaint and legal action filed": [0.0, 0.0, 0.0],
            "settlement agreement after class action litigation": [0.0, 0.0, 0.0],
            "company files for bankruptcy under chapter 11": [0.0, 0.0, 0.0],
            "insolvency, debt restructuring, and creditor negotiations": [0.0, 0.0, 0.0],
            "liquidation or default after financial distress": [0.0, 0.0, 0.0],
            "general corporate news without a clear catalyst": [0.0, 0.0, 0.0],
            "miscellaneous financial news item": [0.0, 0.0, 0.0],
            "other market-related headline": [0.0, 0.0, 0.0],
        }

    async def embed_text(self, text: str) -> FakeEmbeddingResult:
        return FakeEmbeddingResult(self.vector_map.get(text.lower(), [0.0, 0.0, 0.0]))

    async def embed_events(self, events):
        return [await self.embed_text(event.title) for event in events]


@pytest.mark.asyncio
async def test_classifier_detects_earnings():
    classifier = EventClassifier(embedding_service=FakeEmbeddingService())

    result = await classifier.classify("Company reports record revenue and raises full year guidance")

    assert result["event_type"] == "earnings"
    assert 0.0 <= result["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_classifier_detects_merger_acquisition():
    classifier = EventClassifier(embedding_service=FakeEmbeddingService())

    result = await classifier.classify("Apple acquires AI startup")

    assert result["event_type"] == "merger_acquisition"
    assert 0.0 <= result["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_classifier_embedding_fallback_handles_synonyms():
    classifier = EventClassifier(embedding_service=FakeEmbeddingService())

    result = await classifier.classify("Top executive exits in management shakeup")

    assert result["event_type"] == "leadership_change"
    assert 0.0 <= result["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_classifier_returns_other_for_unrelated_headline():
    classifier = EventClassifier(embedding_service=FakeEmbeddingService())

    result = await classifier.classify("Stocks edge higher as markets await data")

    assert result["event_type"] == "other"
    assert 0.0 <= result["confidence"] <= 1.0
