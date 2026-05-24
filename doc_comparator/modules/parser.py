"""
Phase 1: Document Ingestion & Section-Aware Parser
====================================================
Handles PDF/text extraction and splits 10-K documents into
structured sections by Item number (Item 1, 1A, 7, 7A, etc.)

Supports:
  - PDF input (via PyMuPDF)
  - Plain text input (.txt)
  - Auto-detection of 10-K item boundaries
  - Fallback chunking for non-standard documents
"""

import re
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class Section:
    item_id: str          # e.g. "1A", "7", "7A"
    title: str            # e.g. "Risk Factors"
    content: str          # raw text of the section
    page_start: int       # approximate page number (0 if unknown)
    word_count: int       # pre-computed for token cost estimation
    char_count: int

@dataclass
class ParsedDocument:
    filename: str
    total_pages: int
    total_words: int
    sections: list[Section]
    unparsed_preamble: str  # text before Item 1 (cover page, table of contents)
    parse_method: str       # "pdf" | "text"


# ── Known 10-K section map ────────────────────────────────────────────────────

# SEC standard 10-K items. We'll detect these by regex.
KNOWN_ITEMS = {
    "1":   "Business",
    "1A":  "Risk Factors",
    "1B":  "Unresolved Staff Comments",
    "1C":  "Cybersecurity",
    "2":   "Properties",
    "3":   "Legal Proceedings",
    "4":   "Mine Safety Disclosures",
    "5":   "Market for Registrant's Common Equity",
    "6":   "Selected Financial Data",
    "7":   "Management's Discussion and Analysis",
    "7A":  "Quantitative and Qualitative Disclosures About Market Risk",
    "8":   "Financial Statements and Supplementary Data",
    "9":   "Changes in and Disagreements With Accountants",
    "9A":  "Controls and Procedures",
    "9B":  "Other Information",
    "10":  "Directors, Executive Officers and Corporate Governance",
    "11":  "Executive Compensation",
    "12":  "Security Ownership",
    "13":  "Certain Relationships and Related Transactions",
    "14":  "Principal Accountant Fees and Services",
    "15":  "Exhibits and Financial Statement Schedules",
}

# Regex: matches "Item 1.", "ITEM 1A.", "Item 7A" etc. at the start of a line
# Allows for bold/all-caps formatting variants found in real filings
ITEM_PATTERN = re.compile(
    r'^\s*(?:ITEM|Item)\s+'           # "ITEM " or "Item "
    r'(\d{1,2}[AB]?)'                 # capture: "1", "1A", "7A", "15" etc.
    r'[\.\:\s]'                       # separator: period, colon, or space
    r'(.{0,80}?)$',                   # optional title text on same line (≤80 chars)
    re.MULTILINE
)


# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf(path: Path) -> tuple[str, int]:
    """
    Extract full text from a PDF using PyMuPDF (fitz).
    Returns (full_text, page_count).

    PyMuPDF preserves reading order better than pdfplumber for
    dense financial documents. It handles multi-column layouts and
    footnotes reasonably well.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF not installed. Run: pip install pymupdf"
        )

    doc = fitz.open(str(path))
    pages = []
    for page_num, page in enumerate(doc):
        # 'text' mode preserves newlines; 'blocks' mode is better for tables
        # We use 'text' here — tables in 10-Ks are less critical than narrative
        text = page.get_text("text")
        pages.append(text)

    full_text = "\n".join(pages)
    page_count = len(doc)
    doc.close()
    return full_text, page_count


def extract_text_from_txt(path: Path) -> tuple[str, int]:
    """Read plain text file. Page count estimated at 500 words/page."""
    text = path.read_text(encoding="utf-8", errors="replace")
    word_count = len(text.split())
    estimated_pages = max(1, word_count // 500)
    return text, estimated_pages


# ── Section splitter ──────────────────────────────────────────────────────────

def split_into_sections(full_text: str) -> tuple[str, list[Section]]:
    """
    Core algorithm: split document text into 10-K sections.

    Strategy:
    1. Find all ITEM boundary positions using regex
    2. Use those positions as section start/end markers
    3. Assign known titles; fall back to extracted title text
    4. Handle edge cases: duplicate item numbers (TOC matches), 
       items out of order, missing items

    Returns:
        preamble: text before the first detected Item
        sections: ordered list of Section objects
    """
    matches = list(ITEM_PATTERN.finditer(full_text))

    if not matches:
        # Fallback: no standard items found — treat whole doc as one section
        print("  ⚠  No standard Item headings found. Using fallback chunking.")
        return _fallback_chunking(full_text)

    # Filter out table-of-contents hits:
    # Real section boundaries appear once; TOC lists them all upfront.
    # Heuristic: if two matches with the same item_id appear within 3000 chars,
    # the first is a TOC entry — skip it.
    filtered = _deduplicate_toc_matches(matches, full_text)

    if len(filtered) < 2:
        print("  ⚠  Too few sections detected after TOC filtering. Using fallback.")
        return _fallback_chunking(full_text)

    preamble = full_text[:filtered[0].start()].strip()
    sections = []

    for i, match in enumerate(filtered):
        item_id = match.group(1).strip()
        raw_title = match.group(2).strip()

        # Prefer the canonical title from our map; fall back to extracted text
        title = KNOWN_ITEMS.get(item_id, raw_title or f"Item {item_id}")

        # Content runs from this match to the next (or end of doc)
        content_start = match.end()
        content_end = filtered[i + 1].start() if i + 1 < len(filtered) else len(full_text)
        content = full_text[content_start:content_end].strip()

        # Estimate page number from character position
        chars_per_page = max(1, len(full_text) // max(1, len(filtered)))
        page_start = match.start() // chars_per_page

        word_count = len(content.split())
        sections.append(Section(
            item_id=item_id,
            title=title,
            content=content,
            page_start=page_start,
            word_count=word_count,
            char_count=len(content),
        ))

    return preamble, sections


def _deduplicate_toc_matches(matches, full_text: str):
    """
    Remove TOC entries by keeping only the LAST occurrence of each item_id
    within the first 20% of the document if duplicates exist.

    Real 10-Ks often have:
      - "Item 1A. Risk Factors  ...... 12"  (TOC, page 3)
      - "Item 1A. Risk Factors"              (actual section, page 12)

    We keep the actual section (later occurrence).
    """
    doc_len = len(full_text)
    toc_threshold = doc_len * 0.20  # first 20% is likely TOC region

    seen_items: dict[str, list] = {}
    for m in matches:
        item_id = m.group(1).strip()
        seen_items.setdefault(item_id, []).append(m)

    result = []
    for m in matches:
        item_id = m.group(1).strip()
        occurrences = seen_items[item_id]
        if len(occurrences) > 1 and m.start() < toc_threshold:
            # This is likely a TOC entry — skip it
            continue
        result.append(m)

    # Sort by position to maintain document order
    result.sort(key=lambda m: m.start())
    return result


def _fallback_chunking(full_text: str) -> tuple[str, list[Section]]:
    """
    When no standard Item headings are found, split into ~2000-word chunks.
    Used for non-standard documents or annual reports that don't follow
    the SEC Item numbering format.
    """
    words = full_text.split()
    chunk_size = 2000
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

    sections = []
    for i, chunk_words in enumerate(chunks):
        content = " ".join(chunk_words)
        sections.append(Section(
            item_id=str(i + 1),
            title=f"Chunk {i + 1}",
            content=content,
            page_start=i * 4,  # rough estimate
            word_count=len(chunk_words),
            char_count=len(content),
        ))

    return "", sections


# ── Public API ────────────────────────────────────────────────────────────────

def parse_document(file_path: str) -> ParsedDocument:
    """
    Main entry point. Accepts a PDF or .txt path.
    Returns a fully structured ParsedDocument.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"\n📄 Parsing: {path.name}")

    # Extract raw text
    if path.suffix.lower() == ".pdf":
        full_text, total_pages = extract_text_from_pdf(path)
        parse_method = "pdf"
    elif path.suffix.lower() in (".txt", ".md"):
        full_text, total_pages = extract_text_from_txt(path)
        parse_method = "text"
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .pdf or .txt")

    print(f"   Extracted {len(full_text):,} characters across {total_pages} pages")

    # Split into sections
    preamble, sections = split_into_sections(full_text)

    total_words = sum(s.word_count for s in sections)

    doc = ParsedDocument(
        filename=path.name,
        total_pages=total_pages,
        total_words=total_words,
        sections=sections,
        unparsed_preamble=preamble[:2000],  # keep first 2000 chars of preamble
        parse_method=parse_method,
    )

    # Print summary
    print(f"   ✅ Found {len(sections)} sections | {total_words:,} words total")
    for s in sections:
        print(f"      Item {s.item_id:>3} │ {s.title:<45} │ {s.word_count:>6,} words")

    return doc


def save_parsed(doc: ParsedDocument, output_dir: str = "output") -> str:
    """Serialize the ParsedDocument to JSON for downstream phases."""
    out_path = Path(output_dir) / f"{Path(doc.filename).stem}_parsed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # asdict handles nested dataclasses
    data = asdict(doc)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\n💾 Saved parsed document → {out_path}")
    return str(out_path)


def load_parsed(json_path: str) -> ParsedDocument:
    """Load a previously saved ParsedDocument from JSON."""
    data = json.loads(Path(json_path).read_text())
    data["sections"] = [Section(**s) for s in data["sections"]]
    return ParsedDocument(**data)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parser.py <file.pdf> [output_dir]")
        print("       python parser.py doc_a.pdf doc_b.pdf")
        sys.exit(1)

    output_dir = "output"
    files = sys.argv[1:]

    for f in files:
        doc = parse_document(f)
        save_parsed(doc, output_dir)
