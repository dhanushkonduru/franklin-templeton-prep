"""LLM-friendly schema formatter optimized for token-constrained models.

Purpose:
- Convert SQLAlchemy introspection output into compact, readable summaries.
- Produce per-table documents suitable for semantic retrieval (embeddings).
- Avoid token bloat by using compact column notation and short natural-language
  one-line descriptions.

Usage:
  from sql_finance_copilot.schema.llm_formatter import (
      build_prompt_snippet, generate_embedding_docs
  )

  prompt = build_prompt_snippet(schema_dict)
  docs = generate_embedding_docs(schema_dict)
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List

LOG = logging.getLogger("llm_formatter")


def _col_compact(c: Dict[str, Any]) -> str:
    """Return a compact representation for a column: name:type[PK][FK->t.c]."""
    parts = [c.get("name") or "?"]
    typ = c.get("type")
    if typ:
        parts.append(typ.split("(")[0])  # strip precision
    flags = []
    if c.get("primary_key"):
        flags.append("PK")
    fks = c.get("foreign_keys") or []
    if fks:
        fk_targets = ",".join([f"{fk0['referred_table']}.{fk0['referred_column']}" for fk0 in fks])
        flags.append(f"FK->{fk_targets}")
    if flags:
        parts.append("[" + ",".join(flags) + "]")
    return ":".join(parts)


def _table_compact(table: str, meta: Dict[str, Any], max_cols: int = 10) -> str:
    """Compact one-line table summary: name(col:type[flags],...)

    Truncates column list if too long.
    """
    cols = meta.get("columns", [])
    col_texts = [_col_compact(c) for c in cols[:max_cols]]
    if len(cols) > max_cols:
        col_texts.append("...")
    return f"{table}({', '.join(col_texts)})"


def build_prompt_snippet(schema_dict: Dict[str, Any], max_chars: int = 1200) -> str:
    """Build a concise schema snippet suitable to include in an LLM prompt.

    - Uses compact table summaries separated by semicolons.
    - Stops when `max_chars` is reached to avoid token bloat.
    """
    tables = list(schema_dict.get("tables", {}).items())
    parts: List[str] = []
    total = 0
    for table, meta in tables:
        t = _table_compact(table, meta, max_cols=8)
        if total + len(t) + 2 > max_chars:
            break
        parts.append(t)
        total += len(t) + 2

    header = "Schema summary (compact):"
    return header + "\n" + ";\n".join(parts)


def generate_table_doc(table: str, meta: Dict[str, Any], max_chars: int = 250) -> str:
    """Generate a short natural-language one-line doc for semantic retrieval.

    Example: "stocks — security metadata (ticker,name,sector). PK stock_id."""
    cols = meta.get("columns", [])
    names = [c["name"] for c in cols[:6]]
    short_cols = ",".join(names)
    pk = meta.get("primary_key") or []
    pk_txt = f"PK: {','.join(pk)}" if pk else ""
    fk_count = sum(1 for c in cols if c.get("foreign_keys"))
    fk_txt = f"FKs:{fk_count}" if fk_count else ""
    doc = f"{table} — cols: {short_cols}. {pk_txt} {fk_txt}".strip()
    if len(doc) > max_chars:
        return doc[: max_chars - 3] + "..."
    return doc


def generate_embedding_docs(schema_dict: Dict[str, Any], max_chars_per_doc: int = 250) -> List[Dict[str, str]]:
    """Return a list of compact documents for each table suitable for embedding.

    Each doc is a dict: {"id": <table>, "text": <short doc>}.
    """
    docs: List[Dict[str, str]] = []
    for table, meta in schema_dict.get("tables", {}).items():
        text = generate_table_doc(table, meta, max_chars=max_chars_per_doc)
        docs.append({"id": table, "text": text})
    LOG.info("Generated %d embedding docs", len(docs))
    return docs


def format_for_llm(schema_dict: Dict[str, Any], table_limit: int = 20) -> Dict[str, Any]:
    """Return a minimal structured payload optimized for LLMs and retrieval.

    Payload example:
      {
        "tables": {
          "stocks": {"summary": "stocks(...)", "cols": ["stock_id:INTEGER[PK]", ...]},
          ...
        },
        "prompt_snippet": "...",
        "embedding_docs": [{"id":...,"text":...}, ...]
      }
    """
    out: Dict[str, Any] = {"tables": {}}
    tables = list(schema_dict.get("tables", {}).items())[:table_limit]
    for table, meta in tables:
        cols = [_col_compact(c) for c in meta.get("columns", [])]
        out["tables"][table] = {
            "summary": _table_compact(table, meta, max_cols=8),
            "cols": cols,
        }

    out["prompt_snippet"] = build_prompt_snippet(schema_dict)
    out["embedding_docs"] = generate_embedding_docs(schema_dict)
    return out


__all__ = [
    "build_prompt_snippet",
    "generate_embedding_docs",
    "format_for_llm",
]
