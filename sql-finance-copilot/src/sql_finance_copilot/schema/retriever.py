from __future__ import annotations

from sqlalchemy.engine import Engine

from sql_finance_copilot.config import AppSettings
from sql_finance_copilot.core.models import RetrievedSchemaContext
from sql_finance_copilot.db.introspection import SchemaIntrospector
from sql_finance_copilot.schema.embeddings import SentenceTransformerEmbedder
from sql_finance_copilot.schema.index import SchemaIndex


class SchemaRetriever:
    def __init__(self, engine: Engine, settings: AppSettings):
        self._engine = engine
        self._settings = settings
        self._introspector = SchemaIntrospector(engine, settings.allowed_schemas)
        self._embedder = SentenceTransformerEmbedder(settings.embedding_model)
        self._index = SchemaIndex(settings.faiss_index_path, self._embedder)

    def initialize(self, force_rebuild: bool = False) -> None:
        self._settings.ensure_paths()
        if not force_rebuild and self._index.load():
            return
        documents = self._introspector.build_documents(self._settings.max_schema_columns_per_table)
        if not documents:
            raise RuntimeError("No schema documents were discovered. Check allowed schemas and database permissions.")
        self._index.build(documents)

    def retrieve(self, question: str, top_k: int | None = None) -> RetrievedSchemaContext:
        if not self._index.ready:
            self.initialize()
        documents = self._index.search(question, top_k or self._settings.max_schema_tables)
        return RetrievedSchemaContext(question=question, documents=documents)
