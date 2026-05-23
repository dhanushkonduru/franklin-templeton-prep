from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable


logger = logging.getLogger(__name__)


async def async_retry(
    operation: Callable[[], Awaitable[object]],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    operation_name: str = "operation",
) -> object:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as exc:  # pragma: no cover - intentional recovery path
            last_error = exc
            if attempt == attempts:
                break
            retry_after = getattr(exc, "retry_after_seconds", None)
            delay = float(retry_after) if retry_after is not None else min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0.0, delay * 0.15)
            logger.warning(
                "retrying %s after error on attempt %s/%s",
                operation_name,
                attempt,
                attempts,
                extra={"operation": operation_name, "attempt": attempt, "attempts": attempts, "delay_seconds": round(delay, 3), "error": str(exc)},
            )
            await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error
