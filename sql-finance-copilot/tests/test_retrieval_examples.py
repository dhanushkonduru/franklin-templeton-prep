from __future__ import annotations

import numpy as np

from sql_finance_copilot.config import AppSettings
from sql_finance_copilot.core.models import RetrievedSchemaContext, SchemaDocument
from sql_finance_copilot.schema.index import SchemaIndex


class DeterministicEmbedder:
    """Tiny deterministic embedder for retrieval tests without model downloads."""

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            t = text.lower()
            vectors.append(
                [
                    1.0 if "revenue" in t else 0.0,
                    1.0 if "rating" in t else 0.0,
                    1.0 if "price" in t else 0.0,
                ]
            )
        arr = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def test_schema_index_retrieval_top_k(tmp_path):
    docs = [
        SchemaDocument(doc_id="public.financials", schema_name="public", table_name="financials", text="revenue net_income eps"),
        SchemaDocument(doc_id="public.analyst_ratings", schema_name="public", table_name="analyst_ratings", text="rating target_price"),
        SchemaDocument(doc_id="public.daily_prices", schema_name="public", table_name="daily_prices", text="price close volume"),
    ]
    index = SchemaIndex(index_path=tmp_path / "schema_idx", embedder=DeterministicEmbedder())
    index.build(docs)

    hits = index.search("show revenue growth", top_k=2)
    names = [h.table_name for h in hits]

    assert "financials" in names
    assert len(hits) == 2


def test_retrieved_schema_context_properties():
    context = RetrievedSchemaContext(
        question="show ratings",
        documents=[
            SchemaDocument(
                doc_id="public.analyst_ratings",
                schema_name="public",
                table_name="analyst_ratings",
                text="table public.analyst_ratings; columns: rating, target_price",
            )
        ],
    )
    assert context.table_names == ["analyst_ratings"]
    assert context.qualified_table_names == ["public.analyst_ratings"]
    assert "columns" in context.prompt_text
