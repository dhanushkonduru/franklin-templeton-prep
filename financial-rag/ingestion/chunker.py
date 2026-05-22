"""
Chunker
- Tables: never split (kept as-is from parser)
- Text: RecursiveCharacterTextSplitter (fast, reliable)
All metadata (source, page, company, year) is preserved on every chunk.
"""

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from config import get_settings

settings = get_settings()

SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split text documents into chunks. Table documents are kept whole.
    Returns a flat list of chunks with full metadata.
    """
    splitter = build_splitter()

    table_docs  = [d for d in docs if d.metadata.get("chunk_type") == "table"]
    text_docs   = [d for d in docs if d.metadata.get("chunk_type") != "table"]

    logger.info(f"Chunking {len(text_docs)} text docs + keeping {len(table_docs)} tables whole")

    # Split text docs
    text_chunks = splitter.split_documents(text_docs)

    # Tag every chunk with a unique id for citation
    all_chunks = table_docs + text_chunks
    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_id"] = i
        # Ensure page is always present
        chunk.metadata.setdefault("page", 0)

    logger.success(f"Total chunks: {len(all_chunks)} "
                   f"(tables: {len(table_docs)}, text: {len(text_chunks)})")
    return all_chunks


def preview_chunks(chunks: list[Document], n: int = 3) -> None:
    """Print first n chunks for debugging."""
    for i, c in enumerate(chunks[:n]):
        print(f"\n── Chunk {i} ──────────────────────────")
        print(f"Source : {c.metadata.get('source')} | Page {c.metadata.get('page')}")
        print(f"Type   : {c.metadata.get('chunk_type')} | Company: {c.metadata.get('company')}")
        print(f"Length : {len(c.page_content)} chars")
        print(c.page_content[:300], "...")
