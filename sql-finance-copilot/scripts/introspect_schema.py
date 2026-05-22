"""Schema introspection utilities for PostgreSQL using SQLAlchemy.

Produces a structured dictionary suitable for LLM prompt generation and
human-readable formatting. Reusable functions return table lists, column
metadata, primary keys and foreign keys.

Usage:
  python scripts/introspect_schema.py --db-url postgresql://user:pass@localhost:5432/db
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, List, Optional, Any

from sqlalchemy import create_engine, inspect

LOG = logging.getLogger("introspect_schema")


def setup_logging(level: int = logging.INFO) -> None:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(h)


def get_engine(db_url: str):
    return create_engine(db_url)


def list_tables(engine, schema: Optional[str] = None) -> List[str]:
    inspector = inspect(engine)
    tables = inspector.get_table_names(schema=schema)
    LOG.info("Found %d tables in schema=%s", len(tables), schema)
    return tables


def get_table_columns(engine, table_name: str, schema: Optional[str] = None) -> List[Dict[str, Any]]:
    inspector = inspect(engine)
    cols = inspector.get_columns(table_name, schema=schema)
    # Normalize column metadata
    clean_cols = []
    for c in cols:
        clean = {
            "name": c.get("name"),
            "type": str(c.get("type")),
            "nullable": c.get("nullable"),
            "default": c.get("default"),
        }
        clean_cols.append(clean)
    LOG.debug("Columns for %s: %s", table_name, clean_cols)
    return clean_cols


def get_primary_key(engine, table_name: str, schema: Optional[str] = None) -> List[str]:
    inspector = inspect(engine)
    pk = inspector.get_pk_constraint(table_name, schema=schema) or {}
    pk_cols = pk.get("constrained_columns", [])
    LOG.debug("Primary key for %s: %s", table_name, pk_cols)
    return pk_cols


def get_foreign_keys(engine, table_name: str, schema: Optional[str] = None) -> List[Dict[str, Any]]:
    inspector = inspect(engine)
    fks = inspector.get_foreign_keys(table_name, schema=schema)
    # Each fk: {constrained_columns, referred_schema, referred_table, referred_columns, name}
    LOG.debug("Foreign keys for %s: %s", table_name, fks)
    return fks


def introspect_schema(engine, schema: Optional[str] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {"tables": {}}
    tables = list_tables(engine, schema)
    for t in tables:
        cols = get_table_columns(engine, t, schema)
        pk = get_primary_key(engine, t, schema)
        fks = get_foreign_keys(engine, t, schema)

        # Build column map with PK/FK flags
        col_map = []
        fk_map = {}
        for fk in fks:
            for col, ref_col in zip(fk.get("constrained_columns", []), fk.get("referred_columns", [])):
                fk_map.setdefault(col, []).append({
                    "referred_table": fk.get("referred_table"),
                    "referred_schema": fk.get("referred_schema"),
                    "referred_column": ref_col,
                    "constraint_name": fk.get("name"),
                })

        for c in cols:
            c_copy = dict(c)
            c_copy["primary_key"] = c["name"] in pk
            c_copy["foreign_keys"] = fk_map.get(c["name"], [])
            col_map.append(c_copy)

        result["tables"][t] = {
            "columns": col_map,
            "primary_key": pk,
            "foreign_keys": fks,
        }

    return result


def format_for_llm(schema_dict: Dict[str, Any]) -> str:
    """Return a clean, human-readable representation suitable for an LLM prompt.

    The output is a JSON blob with pretty formatting; LLMs can parse either JSON
    or the human-friendly bullet lists here. We include both.
    """
    # JSON section
    json_part = json.dumps(schema_dict, indent=2, default=str)

    # Bullet-list section
    lines = ["Database schema summary:\n"]
    for table, meta in schema_dict.get("tables", {}).items():
        lines.append(f"Table: {table}")
        for col in meta.get("columns", []):
            pk = " PK" if col.get("primary_key") else ""
            fk = ""
            if col.get("foreign_keys"):
                fk_targets = ",".join([f"{fk0['referred_table']}({fk0['referred_column']})" for fk0 in col.get("foreign_keys")])
                fk = f" FK->{fk_targets}"
            lines.append(f"  - {col['name']}: {col['type']}{{nullable={col['nullable']}}}{pk}{fk}")
        lines.append("")

    bullet_part = "\n".join(lines)

    return f"{bullet_part}\n\nJSON:\n{json_part}"


def parse_args():
    p = argparse.ArgumentParser(description="Introspect PostgreSQL schema and format for LLM prompts")
    p.add_argument("--db-url", required=True, help="SQLAlchemy DB URL")
    p.add_argument("--schema", default=None, help="Optional DB schema to inspect (default: search_path)")
    p.add_argument("--out", default=None, help="Optional output file to write the JSON result")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()
    engine = get_engine(args.db_url)
    schema = introspect_schema(engine, args.schema)
    formatted = format_for_llm(schema)
    print(formatted)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(schema, f, default=str, indent=2)
        LOG.info("Wrote schema JSON to %s", args.out)


if __name__ == "__main__":
    main()
