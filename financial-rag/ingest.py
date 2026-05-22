"""
ingest.py  —  Run this ONCE before starting the API.

Steps:
  1. Download 10-K filings from SEC EDGAR        (or skip if files already in data/raw/)
  2. Parse PDFs / HTML → raw LangChain Documents
  3. Chunk documents
  4. Embed + store in ChromaDB

Usage:
  python ingest.py                          # download AAPL, MSFT, TSLA then embed
  python ingest.py --tickers AAPL NVDA      # custom tickers
  python ingest.py --skip-download          # only parse/embed files already in data/raw/
  python ingest.py --preview                # show chunk preview without embedding
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from ingestion.downloader import fetch_tickers
from ingestion.parser import parse_directory
from ingestion.chunker import chunk_documents, preview_chunks
from retrieval.vectorstore import build_vectordb

RAW_DIR = Path("./data/raw")
DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA"]
FILINGS_PER_TICKER = 2


def main():
    parser = argparse.ArgumentParser(description="Financial RAG — ingestion pipeline")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--filings", type=int, default=FILINGS_PER_TICKER)
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, parse files already in data/raw/")
    parser.add_argument("--preview", action="store_true",
                        help="Show chunk preview without embedding")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Financial RAG — Ingestion Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Download ──────────────────────────────────────────
    if not args.skip_download:
        logger.info(f"Step 1: Downloading 10-K filings for {args.tickers}")
        paths = fetch_tickers(args.tickers, filings_per_ticker=args.filings)
        if not paths:
            logger.warning("No files downloaded. Check your internet connection.")
            logger.info("Checking if data/raw/ already has files…")
    else:
        logger.info("Step 1: Skipping download (--skip-download)")

    # Check we have files to work with
    files = list(RAW_DIR.glob("*.*"))
    if not files:
        logger.error(f"No files found in {RAW_DIR}. Run without --skip-download first.")
        sys.exit(1)
    logger.info(f"Files in {RAW_DIR}: {[f.name for f in files]}")

    # ── Step 2: Parse ─────────────────────────────────────────────
    logger.info("Step 2: Parsing documents")
    raw_docs = parse_directory(RAW_DIR)
    if not raw_docs:
        logger.error("No documents parsed. Check file formats (PDF or HTML only).")
        sys.exit(1)
    logger.info(f"Parsed {len(raw_docs)} raw document sections")

    # ── Step 3: Chunk ─────────────────────────────────────────────
    logger.info("Step 3: Chunking")
    chunks = chunk_documents(raw_docs)

    if args.preview:
        logger.info("Preview mode — showing first 5 chunks:")
        preview_chunks(chunks, n=5)
        logger.info("Skipping embedding (--preview). Remove flag to embed.")
        return

    # ── Step 4: Embed + Store ─────────────────────────────────────
    logger.info(f"Step 4: Embedding {len(chunks)} chunks → ChromaDB")
    logger.info("(BGE-large downloads ~1.3 GB on first run — subsequent runs are fast)")
    db = build_vectordb(chunks)

    logger.info("=" * 60)
    logger.success("Ingestion complete!")
    logger.success(f"  Documents parsed : {len(raw_docs)}")
    logger.success(f"  Chunks created   : {len(chunks)}")
    logger.success(f"  Vectors stored   : {db._collection.count()}")
    logger.info("Next step: python -m uvicorn api.main:app --reload")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
