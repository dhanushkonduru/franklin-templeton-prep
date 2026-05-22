from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sql_finance_copilot.core.models import SchemaDocument
from sql_finance_copilot.schema.embeddings import SentenceTransformerEmbedder


class SchemaIndex:
    def __init__(self, index_path: Path, embedder: SentenceTransformerEmbedder):
        self._index_path = index_path
        self._embedder = embedder
        self._index = None
        self._documents: list[SchemaDocument] = []

    @property
    def ready(self) -> bool:
        return self._index is not None and bool(self._documents)

    def build(self, documents: list[SchemaDocument]) -> None:
        if not documents:
            raise ValueError("Schema index cannot be built from an empty document list")

        import faiss

        self._documents = documents
        embeddings = self._embedder.embed([document.text for document in documents])
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(np.asarray(embeddings, dtype="float32"))
        self._index = index
        self.save()

    def save(self) -> None:
        if self._index is None:
            return
        import faiss

        self._index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path / "schema.faiss"))
        documents_payload = [asdict(document) for document in self._documents]
        (self._index_path / "schema_docs.json").write_text(json.dumps(documents_payload, indent=2), encoding="utf-8")

    def load(self) -> bool:
        import faiss

        index_file = self._index_path / "schema.faiss"
        docs_file = self._index_path / "schema_docs.json"
        if not index_file.exists() or not docs_file.exists():
            return False
        self._index = faiss.read_index(str(index_file))
        payload = json.loads(docs_file.read_text(encoding="utf-8"))
        self._documents = [SchemaDocument(**item) for item in payload]
        return True

    def search(self, query: str, top_k: int = 5) -> list[SchemaDocument]:
        if not self.ready:
            raise RuntimeError("Schema index is not loaded")
        query_embedding = self._embedder.embed([query])
        scores, indices = self._index.search(query_embedding, top_k)
        _ = scores
        results: list[SchemaDocument] = []
        for index in indices[0]:
            if index == -1:
                continue
            results.append(self._documents[index])
        return results
