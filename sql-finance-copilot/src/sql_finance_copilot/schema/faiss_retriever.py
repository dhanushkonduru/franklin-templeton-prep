"""Schema retrieval using sentence-transformers and FAISS.

Provides:
- building a compact, token-efficient doc set from schema metadata (supports chunking)
- embedding with SentenceTransformer (small, open-weight models recommended)
- FAISS IndexFlatIP search over normalized vectors
- lightweight reranking that boosts column-token overlap to improve SQL accuracy
- persistence (index + metadata)

Design choices:
- Use compact per-table docs by default (short, one-line descriptions) to minimize token usage.
- Optionally chunk large tables into column-group docs to improve recall for narrow queries.
- Normalize embeddings to unit vectors and use inner-product (IP) for cosine similarity.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

LOG = logging.getLogger("schema_faiss_retriever")


def _simple_tokenize(text: str) -> List[str]:
    # aggressive but small: split on non-alphanum and lowercase
    return [t for t in re.split(r"[^0-9a-zA-Z_]+", text.lower()) if t]


class SchemaFaissRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """Initialize model only; index may be built or loaded later.

        Choose a small embedding model for open-weight efficiency and low token usage.
        """
        self.model_name = model_name
        self.device = device
        LOG.info("Loading embedding model: %s on %s", model_name, device)
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.id_to_doc: Dict[int, Dict] = {}
        self.dim = None

    def _encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # normalize to unit vectors for cosine similarity via inner product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        return vecs.astype("float32")

    def build_index(self, docs: List[Dict[str, str]], save_dir: str | None = None) -> None:
        """Build FAISS index from docs: list of {id: str, text: str}.

        The integer internal ids are assigned starting at 0.
        If `save_dir` is provided, index and metadata are persisted.
        """
        texts = [d["text"] for d in docs]
        ids = [d["id"] for d in docs]
        LOG.info("Encoding %d docs", len(texts))
        vecs = self._encode(texts)
        n, dim = vecs.shape
        self.dim = dim
        LOG.info("Embedding dim=%d, docs=%d", dim, n)

        # create FAISS index (inner product) — supports cosine when vectors are normalized
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        self.index = index

        # store metadata
        self.id_to_doc = {i: {"id": ids[i], "text": texts[i]} for i in range(len(ids))}

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            faiss.write_index(self.index, os.path.join(save_dir, "faiss.index"))
            with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(self.id_to_doc, f, ensure_ascii=False, indent=2)
            # store vectors for debugging/inspection
            np.save(os.path.join(save_dir, "vectors.npy"), vecs)
            LOG.info("Saved index to %s", save_dir)

    def load_index(self, save_dir: str) -> None:
        path_idx = os.path.join(save_dir, "faiss.index")
        path_meta = os.path.join(save_dir, "metadata.json")
        if not os.path.exists(path_idx) or not os.path.exists(path_meta):
            raise FileNotFoundError("Index or metadata not found in %s" % save_dir)
        self.index = faiss.read_index(path_idx)
        with open(path_meta, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # keys were stored as strings of integers
        self.id_to_doc = {int(k): v for k, v in raw.items()}
        # infer dim
        self.dim = self.index.d
        LOG.info("Loaded index (dim=%d) with %d entries", self.dim, len(self.id_to_doc))

    def search(self, query: str, top_k: int = 10, rerank_top_n: int = 50, weight_overlap: float = 3.0) -> List[Dict]:
        """Search for top-k docs for `query`.

        Steps:
        1. Embed query and get top `rerank_top_n` candidates by cosine (inner product).
        2. Rerank candidates by combined score: sim + weight_overlap * token_overlap.
        3. Return top `top_k` items with id, text, sim, overlap, final_score.

        `weight_overlap` boosts docs that share column/ticker tokens with the query — helps SQL accuracy.
        """
        if self.index is None:
            raise RuntimeError("Index not built or loaded")
        qv = self._encode([query])  # shape (1, dim)
        D, I = self.index.search(qv, rerank_top_n)
        sims = D[0]  # inner product (cosine)
        ids = I[0]

        # compute simple token-overlap reranker
        q_tokens = set(_simple_tokenize(query))
        candidates = []
        for idx, sim in zip(ids, sims):
            if idx < 0:
                continue
            meta = self.id_to_doc.get(int(idx), {})
            text = meta.get("text", "")
            doc_tokens = set(_simple_tokenize(text))
            overlap = len(q_tokens & doc_tokens)
            # normalize overlap by sqrt(len(doc_tokens)) to avoid favoring very long docs
            norm = math.sqrt(max(1, len(doc_tokens)))
            overlap_score = overlap / norm
            final = float(sim) + weight_overlap * overlap_score
            candidates.append(
                {
                    "internal_id": int(idx),
                    "id": meta.get("id"),
                    "text": text,
                    "sim": float(sim),
                    "overlap": overlap,
                    "overlap_score": overlap_score,
                    "score": final,
                }
            )

        # sort by combined score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # detection of retrieval failure: if top sim is very low, signal none
        if candidates:
            top_sim = candidates[0]["sim"]
            if top_sim < 0.10:
                LOG.warning(
                    "Low top similarity (%.3f). Retrieval may have failed or query is out-of-domain.", top_sim
                )

        return candidates[:top_k]


def build_docs_from_schema(schema_dict: Dict[str, Any], cols_per_chunk: int = 8) -> List[Dict[str, str]]:
    """Create compact docs from a schema dict (as produced by introspect_schema).

    Strategy:
    - For each table produce one compact summary doc with up to `cols_per_chunk` columns displayed.
    - If a table has more columns than `cols_per_chunk`, produce additional chunk docs with the next group of columns.
    - Doc ID format: "<table>" for primary, "<table>::chunk:<i>" for chunks.
    This keeps per-doc token count low and improves recall for narrow column-specific queries.
    """
    docs: List[Dict[str, str]] = []
    for table, meta in schema_dict.get("tables", {}).items():
        cols = meta.get("columns", [])
        # compact column list: name:type flags
        compact_cols = [
            f"{c['name']}:{str(c.get('type', '')).split('(')[0]}" +
            ("[PK]" if c.get("primary_key") else "")
            for c in cols
        ]
        # primary doc
        primary_text = f"{table} | {', '.join(compact_cols[:cols_per_chunk])}"
        docs.append({"id": table, "text": primary_text})

        # chunk remaining columns
        i = 1
        for start in range(cols_per_chunk, len(compact_cols), cols_per_chunk):
            chunk_cols = compact_cols[start : start + cols_per_chunk]
            text = f"{table}::chunk:{i} | {', '.join(chunk_cols)}"
            docs.append({"id": f"{table}::chunk:{i}", "text": text})
            i += 1

    LOG.info("Built %d retrieval docs from schema (cols_per_chunk=%d)", len(docs), cols_per_chunk)
    return docs


__all__ = [
    "SchemaFaissRetriever",
    "build_docs_from_schema",
]
