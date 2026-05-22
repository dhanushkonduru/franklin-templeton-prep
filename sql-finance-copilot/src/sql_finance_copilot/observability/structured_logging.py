"""Structured logging helpers that reuse the project's logging_config."""

from __future__ import annotations

from sql_finance_copilot.logging_config import configure_logging, request_id_var

# Re-export for convenience
__all__ = ["configure_logging", "request_id_var"]
