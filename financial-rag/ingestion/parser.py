"""
Document parser
Handles both PDF (via pdfplumber + unstructured) and HTML 10-K filings.
Tables are extracted separately and kept as single chunks (never split).
"""

import re
from pathlib import Path
from typing import Generator

import pdfplumber
from langchain.schema import Document
from loguru import logger
from tqdm import tqdm

try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.html import partition_html
    HAS_UNSTRUCTURED = True
except ImportError:
    HAS_UNSTRUCTURED = False
    logger.warning("unstructured not available — falling back to pdfplumber only")


# ── Helpers ────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Normalise whitespace and remove boilerplate artefacts."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?i)table of contents.*?\n", "", text)
    return text.strip()


def _table_to_markdown(table: list[list]) -> str:
    """Convert a pdfplumber table to markdown."""
    if not table or not table[0]:
        return ""
    def cell(c): return str(c).strip() if c else ""
    header = "| " + " | ".join(cell(c) for c in table[0]) + " |"
    sep    = "| " + " | ".join("---" for _ in table[0]) + " |"
    rows   = ["| " + " | ".join(cell(c) for c in row) + " |" for row in table[1:] if any(c for c in row)]
    return "\n".join([header, sep] + rows)


def _infer_company(path: Path) -> str:
    return path.name.split("_")[0].upper()


def _infer_year(path: Path) -> str:
    # filename pattern: TICKER_2023-10-27_...
    parts = path.stem.split("_")
    for p in parts:
        if re.match(r"\d{4}", p):
            return p[:4]
    return "unknown"


# ── PDF Parser ─────────────────────────────────────────────────────────────

def parse_pdf(path: Path) -> list[Document]:
    """
    Extract text and tables from a PDF.
    Returns one Document per page (text) + one Document per table.
    """
    docs: list[Document] = []
    company = _infer_company(path)
    year = _infer_year(path)

    base_meta = {
        "source": path.name,
        "company": company,
        "fiscal_year": year,
        "doc_type": "10-K",
    }

    logger.info(f"Parsing PDF: {path.name}")
    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(tqdm(pdf.pages, desc=f"  {path.name}", leave=False), start=1):
                # ── Tables first (kept whole, tagged as table)
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    md = _table_to_markdown(table)
                    if md and len(md) > 50:
                        docs.append(Document(
                            page_content=f"[TABLE]\n{md}",
                            metadata={**base_meta, "page": page_num, "chunk_type": "table", "table_index": t_idx},
                        ))

                # ── Body text (exclude table bounding boxes to avoid duplication)
                text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                text = _clean(text)
                if len(text) > 100:
                    docs.append(Document(
                        page_content=text,
                        metadata={**base_meta, "page": page_num, "chunk_type": "text"},
                    ))
    except Exception as e:
        logger.error(f"pdfplumber failed on {path.name}: {e}")
        # Fallback to unstructured if available
        if HAS_UNSTRUCTURED:
            docs = _parse_pdf_unstructured(path, base_meta)

    logger.success(f"  → {len(docs)} raw documents from {path.name}")
    return docs


def _parse_pdf_unstructured(path: Path, base_meta: dict) -> list[Document]:
    """Fallback parser using the unstructured library."""
    docs = []
    try:
        elements = partition_pdf(str(path), strategy="fast")
        for i, el in enumerate(elements):
            text = str(el).strip()
            if len(text) > 80:
                docs.append(Document(
                    page_content=text,
                    metadata={
                        **base_meta,
                        "page": getattr(el.metadata, "page_number", 0),
                        "chunk_type": "text",
                        "element_type": type(el).__name__,
                    },
                ))
    except Exception as e:
        logger.error(f"unstructured also failed: {e}")
    return docs


# ── HTML Parser ─────────────────────────────────────────────────────────────

def parse_html(path: Path) -> list[Document]:
    """Parse an HTML 10-K filing (SEC inline XBRL or plain HTML)."""
    docs: list[Document] = []
    company = _infer_company(path)
    year = _infer_year(path)
    base_meta = {"source": path.name, "company": company, "fiscal_year": year, "doc_type": "10-K"}

    logger.info(f"Parsing HTML: {path.name}")
    html_text = path.read_text(encoding="utf-8", errors="replace")

    if HAS_UNSTRUCTURED:
        try:
            elements = partition_html(text=html_text)
            for el in elements:
                text = _clean(str(el))
                if len(text) > 100:
                    docs.append(Document(
                        page_content=text,
                        metadata={**base_meta, "page": 0, "chunk_type": "text"},
                    ))
            logger.success(f"  → {len(docs)} raw documents from {path.name}")
            return docs
        except Exception as e:
            logger.error(f"unstructured HTML parse failed: {e}")

    # Fallback: regex strip HTML tags
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = _clean(text)
    if text:
        docs.append(Document(
            page_content=text,
            metadata={**base_meta, "page": 0, "chunk_type": "text"},
        ))
    return docs


# ── Main dispatcher ─────────────────────────────────────────────────────────

def parse_file(path: Path) -> list[Document]:
    """Parse any supported file type."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    elif suffix in (".htm", ".html"):
        return parse_html(path)
    else:
        logger.warning(f"Unsupported file type: {path.suffix} — skipping {path.name}")
        return []


def parse_directory(directory: Path) -> list[Document]:
    """Parse all supported files in a directory."""
    files = list(directory.glob("*.*"))
    supported = [f for f in files if f.suffix.lower() in (".pdf", ".htm", ".html")]
    logger.info(f"Found {len(supported)} files to parse in {directory}")
    all_docs: list[Document] = []
    for f in supported:
        all_docs.extend(parse_file(f))
    logger.success(f"Total raw documents: {len(all_docs)}")
    return all_docs
