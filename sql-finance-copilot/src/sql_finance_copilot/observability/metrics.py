"""Simple metrics helpers (placeholders for integration with Prometheus/StatsD)."""

from __future__ import annotations

def increment_metric(name: str, value: int = 1, labels: dict | None = None) -> None:
    # Placeholder: swap with a real metrics client in production.
    return None
