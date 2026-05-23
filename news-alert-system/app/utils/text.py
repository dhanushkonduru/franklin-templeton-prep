from __future__ import annotations

import hashlib
import re


_whitespace = re.compile(r"\s+")
_non_word = re.compile(r"[^\w\s\-\./]")


def normalize_text(value: str) -> str:
    cleaned = _non_word.sub(" ", value.lower())
    return _whitespace.sub(" ", cleaned).strip()


def content_hash(*parts: str) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(normalize_text(part).encode("utf-8"))
        hasher.update(b"|")
    return hasher.hexdigest()


def normalize_ticker(value: str) -> str:
    return value.strip().upper()


def compute_embedding_hash(embedding: list[float]) -> str:
    import struct
    if not embedding:
        return ""
    hasher = hashlib.sha256()
    for value in embedding:
        hasher.update(struct.pack("!f", value))
    return hasher.hexdigest()
