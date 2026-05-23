from __future__ import annotations

import asyncio
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Any

import aiohttp

from app.config import settings
from app.utils.retry import async_retry


@dataclass(slots=True)
class RetryAfterError(RuntimeError):
    message: str
    retry_after_seconds: float

    def __str__(self) -> str:
        return self.message


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        retry_at = parsedate_to_datetime(value)
    except Exception:
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    delta = (retry_at - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, delta)


async def _request_text_or_json(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout_seconds: float | None = None,
    operation_name: str,
) -> Any:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds or settings.http_timeout_seconds)
    try:
        async with session.request(method, url, headers=headers, params=params, timeout=timeout) as response:
            if response.status == 429:
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                raise RetryAfterError(
                    f"rate limited by {url} with status 429",
                    retry_after_seconds=retry_after if retry_after is not None else 1.0,
                )
            if 500 <= response.status < 600:
                raise RuntimeError(f"transient HTTP error {response.status} from {url}")
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "json" in content_type:
                return await response.json()
            return await response.text()
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"timeout while calling {url}") from exc


async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout_seconds: float | None = None,
    attempts: int | None = None,
    operation_name: str = "http_fetch_json",
) -> dict[str, Any]:
    result = await async_retry(
        lambda: _request_text_or_json(
            session,
            "GET",
            url,
            headers=headers,
            params=params,
            timeout_seconds=timeout_seconds,
            operation_name=operation_name,
        ),
        attempts=attempts or settings.max_fetch_retries,
        operation_name=operation_name,
    )
    if not isinstance(result, dict):
        raise TypeError(f"expected JSON object from {url}")
    return result


async def fetch_text(
    session: aiohttp.ClientSession,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout_seconds: float | None = None,
    attempts: int | None = None,
    operation_name: str = "http_fetch_text",
) -> str:
    result = await async_retry(
        lambda: _request_text_or_json(
            session,
            "GET",
            url,
            headers=headers,
            params=params,
            timeout_seconds=timeout_seconds,
            operation_name=operation_name,
        ),
        attempts=attempts or settings.max_fetch_retries,
        operation_name=operation_name,
    )
    if not isinstance(result, str):
        raise TypeError(f"expected text response from {url}")
    return result