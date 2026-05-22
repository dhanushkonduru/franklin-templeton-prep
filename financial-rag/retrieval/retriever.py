"""
Hybrid retriever
BM25 (keyword) + Dense (semantic) → Ensemble → Cohere Rerank
This combination handles both exact financial terms (EBITDA, EPS, ticker)
and semantic similarity (questions about growth, risks, outlook).
"""

from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain.schema import Document
from loguru import logger

from config import get_settings
from retrieval.vectorstore import load_vectordb

settings = get_settings()

_retriever_instance = None


def build_retriever(chunks: list[Document] | None = None) -> ContextualCompressionRetriever:
    """
    Build the full retrieval pipeline.
    If chunks is None, loads them from ChromaDB metadata (for API use).
    """
    global _retriever_instance
    if _retriever_instance is not None:
        return _retriever_instance

    db = load_vectordb()

    # ── Dense retriever (ChromaDB similarity search)
    dense_retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_top_k},
    )

    # ── BM25 retriever (requires the actual chunk texts)
    if chunks is None:
        # Reconstruct from ChromaDB — get all stored docs
        logger.info("Loading all chunks from ChromaDB for BM25 index…")
        results = db._collection.get(include=["documents", "metadatas"])
        chunks = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = settings.retrieval_top_k

    # ── Ensemble (weighted combination)
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[settings.bm25_weight, settings.dense_weight],
    )

    # ── Cohere Reranker (re-scores top-k with cross-attention)
    reranker = CohereRerank(
        model=settings.cohere_rerank_model,
        top_n=settings.rerank_top_n,
        cohere_api_key=settings.cohere_api_key,
    )

    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble,
    )

    logger.success(
        f"Retriever ready — BM25({settings.bm25_weight}) + "
        f"Dense({settings.dense_weight}) → Rerank top {settings.rerank_top_n}"
    )
    _retriever_instance = retriever
    return retriever


def reset_retriever() -> None:
    """Force rebuild on next call (useful after re-ingestion)."""
    global _retriever_instance
    _retriever_instance = None
