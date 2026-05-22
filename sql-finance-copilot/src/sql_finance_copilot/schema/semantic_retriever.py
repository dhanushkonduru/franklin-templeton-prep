"""Semantic schema retriever optimized for Groq prompt usage.

Features:
- Embed compact schema descriptions (per-table docs) with SentenceTransformer
- Index vectors with FAISS (IndexFlatIP on normalized vectors)
- Embed user question, retrieve top candidates, rerank by token overlap
- Return a concise `schema_context` string optimized for inclusion in Groq prompts

Usage example:
  retriever = SemanticRetriever()
  retriever.build_index(docs, save_dir='data/schema_index')
  result = retriever.retrieve_and_format("revenue growth last 3 years", top_k=3)
  print(result['schema_context'])
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Dict, List, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from sql_finance_copilot.schema.llm_formatter import format_for_llm

LOG = logging.getLogger("semantic_retriever")


def _tokenize(s: str) -> List[str]:
    return [t for t in re.split(r"[^0-9a-zA-Z_]+", s.lower()) if t]


class SemanticRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        LOG.info("Loading embedding model %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.Index] = None
        self.id_to_doc: Dict[int, Dict[str, Any]] = {}
        self.dim: Optional[int] = None

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        return vecs.astype("float32")

    def build_index(self, docs: List[Dict[str, str]], save_dir: Optional[str] = None) -> None:
        """Build FAISS index from `docs` where each doc is {id, text, table?}.

        Stores `id_to_doc` metadata mapping internal integer ids to original doc info.
        If `save_dir` is provided the index and metadata are persisted.
        """
        texts = [d["text"] for d in docs]
        ids = [d.get("id") or d.get("table") or str(i) for i, d in enumerate(docs)]
        vecs = self._embed(texts)
        n, dim = vecs.shape
        self.dim = dim
        LOG.info("Built embeddings: n=%d dim=%d", n, dim)

        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        self.index = index

        self.id_to_doc = {i: {"id": ids[i], "text": texts[i], "table": docs[i].get("table")} for i in range(n)}

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            faiss.write_index(self.index, os.path.join(save_dir, "faiss.index"))
            with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(self.id_to_doc, f, ensure_ascii=False, indent=2)
            LOG.info("Saved index and metadata to %s", save_dir)

    def load_index(self, save_dir: str) -> None:
        idx_path = os.path.join(save_dir, "faiss.index")
        meta_path = os.path.join(save_dir, "metadata.json")
        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Index or metadata not found in %s" % save_dir)
        self.index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.id_to_doc = {int(k): v for k, v in json.load(f).items()}
        self.dim = self.index.d
        LOG.info("Loaded index (dim=%d) with %d entries", self.dim, len(self.id_to_doc))

    def retrieve(self, query: str, top_k: int = 5, rerank_top_n: int = 50, overlap_weight: float = 2.0) -> List[Dict[str, Any]]:
        """Retrieve and rerank candidate docs for `query`.

        Returns a list of candidates with fields: id, table, text, sim, overlap, score.
        """
        if self.index is None:
            raise RuntimeError("Index not built or loaded")
        qv = self._embed([query])
        D, I = self.index.search(qv, rerank_top_n)
        sims = D[0]
        ids = I[0]

        q_tokens = set(_tokenize(query))
        candidates: List[Dict[str, Any]] = []
        for idx, sim in zip(ids, sims):
            if idx < 0:
                continue
            meta = self.id_to_doc.get(int(idx), {})
            text = meta.get("text", "")
            table = meta.get("table") or meta.get("id")
            doc_tokens = set(_tokenize(text))
            overlap = len(q_tokens & doc_tokens)
            norm = math.sqrt(max(1, len(doc_tokens)))
            overlap_score = overlap / norm
            score = float(sim) + overlap_weight * overlap_score
            candidates.append({
                "internal_id": int(idx),
                "id": meta.get("id"),
                "table": table,
                "text": text,
                "sim": float(sim),
                "overlap": overlap,
                "overlap_score": overlap_score,
                "score": score,
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        # low-sim detection
        if candidates and candidates[0]["sim"] < 0.10:
            LOG.warning("Low top similarity %.3f — query may be out-of-domain", candidates[0]["sim"])

        return candidates[:top_k]

    def retrieve_and_format(self, schema_dict: Dict[str, Any], query: str, top_k: int = 5) -> Dict[str, Any]:
        """High-level helper: build docs from schema_dict, ensure index present, retrieve top_k, and return formatted context.

        Returns:
          {"tables": [table_names], "schema_context": <compact string>, "hits": [candidates]}
        The `schema_context` is optimized for Groq prompts (short prompt_snippet from `format_for_llm`).
        """
        # If index not built, build from schema_dict docs using llm_formatter's compact docs
        if self.index is None:
            # build docs: use format_for_llm to create compact per-table docs
            payload = format_for_llm(schema_dict)
            docs = payload.get("embedding_docs", [])
            # ensure docs have table/id fields
            docs_prepped = [{"id": d.get("id"), "text": d.get("text"), "table": d.get("id")} for d in docs]
            self.build_index(docs_prepped)

        hits = self.retrieve(query, top_k=top_k)
        table_names = []
        # collect subset of schema for formatting
        subset = {"tables": {}}
        for h in hits:
            t = h.get("table")
            if t and t not in table_names:
                table_names.append(t)
                if t in schema_dict.get("tables", {}):
                    subset["tables"][t] = schema_dict["tables"][t]

        # Format compact schema context using llm_formatter but only for retrieved subset
        formatted = format_for_llm(subset)
        # Groq optimization: return the short `prompt_snippet` which is concise and token-efficient
        schema_context = formatted.get("prompt_snippet")

        return {"tables": table_names, "schema_context": schema_context, "hits": hits}


__all__ = ["SemanticRetriever"]
