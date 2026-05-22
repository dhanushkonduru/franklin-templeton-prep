from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from sql_finance_copilot.core.models import ColumnInfo, SchemaDocument, TableInfo


class SchemaIntrospector:
    def __init__(self, engine: Engine, allowed_schemas: Iterable[str] | None = None):
        self._engine = engine
        # For most Postgres setups use 'public'. For SQLite, reflection must
        # be performed without a schema name (pass None) — detect dialect.
        if engine.dialect.name == "sqlite":
            # inspector.get_table_names(schema=None) will list the sqlite tables
            self._allowed_schemas = [None]
        else:
            self._allowed_schemas = list(allowed_schemas or ["public"])

    def inspect_tables(self) -> list[TableInfo]:
        inspector = inspect(self._engine)
        tables: list[TableInfo] = []
        for schema_name in self._allowed_schemas:
            for table_name in inspector.get_table_names(schema=schema_name):
                columns = [
                    ColumnInfo(
                        name=column_info["name"],
                        data_type=str(column_info.get("type", "unknown")),
                        nullable=bool(column_info.get("nullable", True)),
                        default=column_info.get("default"),
                    )
                    for column_info in inspector.get_columns(table_name, schema=schema_name)
                ]
                try:
                    table_comment = inspector.get_table_comment(table_name, schema=schema_name) or {}
                except NotImplementedError:
                    # Some dialects (e.g., SQLite) do not implement table comments
                    table_comment = {}
                tables.append(
                    TableInfo(
                        schema_name=schema_name,
                        table_name=table_name,
                        columns=columns,
                        comment=table_comment.get("text"),
                    )
                )
        return tables

    def build_documents(self, max_columns_per_table: int = 40) -> list[SchemaDocument]:
        documents: list[SchemaDocument] = []
        for table in self.inspect_tables():
            documents.append(
                SchemaDocument(
                    doc_id=table.qualified_name,
                    schema_name=table.schema_name,
                    table_name=table.table_name,
                    text=table.to_document(max_columns=max_columns_per_table),
                )
            )
        return documents
