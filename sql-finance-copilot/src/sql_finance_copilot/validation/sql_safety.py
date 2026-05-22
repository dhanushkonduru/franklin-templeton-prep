from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from sql_finance_copilot.core.models import ValidationResult


class SQLSafetyError(ValueError):
    pass


@dataclass(slots=True)
class SQLValidationPolicy:
    allowed_tables: set[str] = field(default_factory=set)
    allowed_columns: dict[str, set[str]] = field(default_factory=dict)
    max_rows: int = 100
    reject_comments: bool = True
    safe_functions: set[str] = field(
        default_factory=lambda: {
            # aggregates
            "count",
            "avg",
            "sum",
            "min",
            "max",
            "stddev_samp",
            "stddev_pop",
            "var_samp",
            "var_pop",
            "corr",
            "covar_samp",
            "covar_pop",
            "rank",
            "dense_rank",
            "row_number",
            "percent_rank",
            "percentile_cont",
            "percentile_disc",
            "lag",
            "lead",
            # scalar / date / numeric helpers
            "abs",
            "ceil",
            "ceiling",
            "coalesce",
            "cast",
            "date_trunc",
            "extract",
            "floor",
            "greatest",
            "least",
            "length",
            "lower",
            "ln",
            "log",
            "nullif",
            "now",
            "power",
            "round",
            "sqrt",
            "substring",
            "to_date",
            "to_timestamp",
            "trim",
            "upper",
            "current_date",
            "current_timestamp",
            # logical operators parsed as functions by sqlglot
            "and",
            "or",
            "not",
            # sqlite date/time helper
            "strftime",
            # Note: sqlglot can emit ANONYMOUS for function-like nodes; handled in validator
        }
    )


class SQLSafetyValidator:
    blocked_keywords = re.compile(
        r"\b(insert|update|delete|merge|create|drop|alter|truncate|copy|vacuum|call|grant|revoke|execute|prepare|deallocate|do|lock|analyze|comment|refresh)\b",
        flags=re.IGNORECASE,
    )
    comment_markers = re.compile(r"(--|/\*|\*/|#)")

    def __init__(self, policy: SQLValidationPolicy | None = None):
        self._policy = policy or SQLValidationPolicy()

    def validate(
        self,
        sql: str,
        allowed_tables: list[str],
        allowed_columns: Mapping[str, Sequence[str]] | None = None,
        max_rows: int | None = None,
    ) -> ValidationResult:
        policy = self._policy
        if allowed_columns is not None or allowed_tables:
            policy = SQLValidationPolicy(
                allowed_tables={self._normalize_name(table) for table in allowed_tables},
                allowed_columns=self._normalize_allowed_columns(allowed_columns or {}),
                max_rows=max_rows if max_rows is not None else policy.max_rows,
                reject_comments=policy.reject_comments,
                safe_functions=set(policy.safe_functions),
            )

        normalized_sql = sql.strip().rstrip(";")
        if not normalized_sql:
            raise SQLSafetyError("SQL is empty")

        if policy.reject_comments and self.comment_markers.search(normalized_sql):
            raise SQLSafetyError("SQL comments are not allowed")

        if ";" in normalized_sql:
            raise SQLSafetyError("Multiple statements are not allowed")

        if self.blocked_keywords.search(normalized_sql):
            raise SQLSafetyError("Blocked write or admin keyword detected")

        try:
            statements = sqlglot.parse(normalized_sql, read="postgres")
        except sqlglot.errors.ParseError as exc:
            raise SQLSafetyError(f"SQL parse error: {exc}") from exc

        if len(statements) != 1:
            raise SQLSafetyError("Multiple statements are not allowed")

        parsed = statements[0]
        if not isinstance(parsed, (exp.Select, exp.Union)):
            raise SQLSafetyError("Only read-only SELECT queries are allowed")

        if parsed.find(exp.Into):
            raise SQLSafetyError("SELECT INTO is not allowed")

        self._validate_functions(parsed, policy.safe_functions)

        cte_names = {self._normalize_name(cte.alias_or_name) for cte in parsed.find_all(exp.CTE)}
        alias_to_table, referenced_tables = self._validate_tables(parsed, policy.allowed_tables, cte_names)
        self._validate_columns(parsed, alias_to_table, policy.allowed_columns, referenced_tables)
        parsed = self._enforce_limit(parsed, policy.max_rows)

        return ValidationResult(sql=parsed.sql(dialect="postgres"), referenced_tables=referenced_tables)

    def _validate_functions(self, parsed: exp.Expression, safe_functions: set[str]) -> None:
        for node in parsed.find_all(exp.Func):
            # sqlglot may represent some function calls as 'ANONYMOUS' with the actual
            # function name available in node.args['this'] (either a string or an Identifier).
            raw_name = node.sql_name()
            func_candidate = None
            if raw_name and raw_name.upper() == "ANONYMOUS":
                this = node.args.get("this")
                if isinstance(this, str):
                    func_candidate = this
                elif hasattr(this, "this"):
                    func_candidate = getattr(this, "this")

            func_name = self._normalize_name(func_candidate or raw_name)
            if func_name and func_name not in safe_functions:
                raise SQLSafetyError(f"Function not allowed: {func_name}")

    def _validate_tables(
        self,
        parsed: exp.Expression,
        allowed_tables: set[str],
        cte_names: set[str],
    ) -> tuple[dict[str, str], list[str]]:
        alias_to_table: dict[str, str] = {}
        referenced_tables: list[str] = []

        for table in parsed.find_all(exp.Table):
            table_name = self._normalize_name(table.name)
            schema_name = self._normalize_name(table.db) if table.db else ""
            qualified = f"{schema_name}.{table_name}" if schema_name else table_name

            if table_name in cte_names and not table.find_ancestor(exp.CTE):
                continue

            if qualified not in allowed_tables and table_name not in allowed_tables:
                raise SQLSafetyError(f"Table not allowed: {qualified}")

            canonical = qualified if qualified in allowed_tables else table_name
            referenced_tables.append(canonical)

            alias = self._normalize_name(getattr(table, "alias_or_name", table_name))
            alias_to_table[alias] = canonical
            alias_to_table[table_name] = canonical
            if schema_name:
                alias_to_table[qualified] = canonical

        return alias_to_table, referenced_tables

    def _validate_columns(
        self,
        parsed: exp.Expression,
        alias_to_table: Mapping[str, str],
        allowed_columns: Mapping[str, set[str]],
        referenced_tables: Sequence[str],
    ) -> None:
        if not allowed_columns:
            return

        referenced_table_set = {self._normalize_name(table) for table in referenced_tables}
        allowed_table_keys = set(allowed_columns.keys())

        for column in parsed.find_all(exp.Column):
            column_name = self._normalize_name(column.name)
            if column_name == "*":
                if column.table:
                    table_key = self._resolve_table_key(column.table, alias_to_table, allowed_table_keys)
                    if table_key is None:
                        raise SQLSafetyError(f"Unknown table for wildcard column: {column.table}")
                elif len(referenced_table_set) > 1:
                    raise SQLSafetyError("Bare wildcard SELECT is not allowed across multiple tables")
                continue

            if column.table:
                table_key = self._resolve_table_key(column.table, alias_to_table, allowed_table_keys)
                if table_key is None:
                    raise SQLSafetyError(f"Unknown table or alias for column: {column.table}.{column_name}")
                if column_name not in allowed_columns.get(table_key, set()):
                    raise SQLSafetyError(f"Column not allowed: {table_key}.{column_name}")
                continue

            candidate_tables = [
                table_key
                for table_key in referenced_tables
                if column_name in allowed_columns.get(self._normalize_name(table_key), set())
            ]
            if len(candidate_tables) == 1:
                continue
            if len(candidate_tables) > 1:
                raise SQLSafetyError(f"Ambiguous unqualified column: {column_name}")
            raise SQLSafetyError(f"Column not allowed or unknown: {column_name}")

    def _enforce_limit(self, parsed: exp.Expression, max_rows: int) -> exp.Expression:
        limit = parsed.args.get("limit")
        if limit is None:
            return parsed.limit(max_rows)

        limit_value = self._extract_limit_value(limit)
        if limit_value is None or limit_value > max_rows:
            return parsed.limit(max_rows)
        return parsed

    def _extract_limit_value(self, limit: exp.Expression) -> int | None:
        expression = limit.args.get("expression")
        if isinstance(expression, exp.Literal) and not expression.is_string:
            try:
                return int(expression.this)
            except (TypeError, ValueError):
                return None
        return None

    def _resolve_table_key(
        self,
        table_reference: str,
        alias_to_table: Mapping[str, str],
        allowed_table_keys: set[str],
    ) -> str | None:
        normalized = self._normalize_name(table_reference)
        if normalized in alias_to_table:
            return alias_to_table[normalized]
        if normalized in allowed_table_keys:
            return normalized
        return None

    def _normalize_allowed_columns(self, allowed_columns: Mapping[str, Sequence[str]]) -> dict[str, set[str]]:
        return {
            self._normalize_name(table): {self._normalize_name(column) for column in columns}
            for table, columns in allowed_columns.items()
        }

    def _normalize_name(self, value: str | None) -> str:
        return (value or "").strip().strip('"').lower()
