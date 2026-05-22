"""
Vector store manager
Builds and persists a ChromaDB collection from document chunks.
Uses local HuggingFace BGE embeddings — no API key required.
"""

import os
from pathlib import Path

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from config import get_settings

settings = get_settings()

_embeddings_instance = None
_vectordb_instance = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Singleton — load the embedding model once."""
    global _embeddings_instance
    if _embeddings_instance is None:
        logger.info(f"Loading embedding model: {settings.embedding_model} (first run downloads ~1.3 GB)")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.success("Embedding model loaded")
    return _embeddings_instance


def build_vectordb(chunks: list[Document]) -> Chroma:
    """Embed chunks and persist to ChromaDB. Overwrites any existing collection."""
    embeddings = get_embeddings()
    persist_dir = settings.chroma_persist_dir
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Embedding {len(chunks)} chunks → ChromaDB at {persist_dir}")
    logger.info("This takes 2–10 min on CPU depending on document count…")

    # Batch to avoid memory spikes
    BATCH = 64
    if len(chunks) <= BATCH:
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name=settings.chroma_collection,
        )
    else:
        # Create with first batch, then add rest
        db = Chroma.from_documents(
            documents=chunks[:BATCH],
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name=settings.chroma_collection,
        )
        for start in range(BATCH, len(chunks), BATCH):
            batch = chunks[start:start + BATCH]
            db.add_documents(batch)
            logger.debug(f"  Embedded {min(start + BATCH, len(chunks))}/{len(chunks)} chunks")

    count = db._collection.count()
    logger.success(f"ChromaDB ready — {count} vectors in '{settings.chroma_collection}'")
    return db


def load_vectordb() -> Chroma:
    """Load an existing ChromaDB collection from disk."""
    global _vectordb_instance
    if _vectordb_instance is None:
        persist_dir = settings.chroma_persist_dir
        if not Path(persist_dir).exists():
            raise FileNotFoundError(
                f"No ChromaDB found at {persist_dir}. Run ingestion first:\n"
                "  python ingest.py"
            )
        logger.info(f"Loading ChromaDB from {persist_dir}")
        _vectordb_instance = Chroma(
            persist_directory=persist_dir,
            embedding_function=get_embeddings(),
            collection_name=settings.chroma_collection,
        )
        count = _vectordb_instance._collection.count()
        logger.success(f"Loaded ChromaDB — {count} vectors")
    return _vectordb_instance


def collection_stats() -> dict:
    db = load_vectordb()
    count = db._collection.count()
    return {"total_vectors": count, "collection": settings.chroma_collection, "persist_dir": settings.chroma_persist_dir}
