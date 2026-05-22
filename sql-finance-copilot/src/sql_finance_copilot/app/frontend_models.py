from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class QueryHistoryItem:
    asked_at: datetime
    question: str
    elapsed_ms: float
    row_count: int
    repaired: bool
    repair_attempts: int
    error: str | None
    sql: str
    validated_sql: str
    rows_preview: list[dict[str, Any]] = field(default_factory=list)
