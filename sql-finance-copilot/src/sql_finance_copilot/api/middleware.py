from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from sql_finance_copilot.config import AppSettings
from sql_finance_copilot.logging_config import request_id_var

LOG = logging.getLogger("api.request")


def install_middleware(app: FastAPI, settings: AppSettings) -> None:
    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        token = request_id_var.set(str(uuid.uuid4()))
        started = time.perf_counter()
        try:
            if request.method in {"POST", "PUT", "PATCH"}:
                content_length = request.headers.get("content-length")
                try:
                    if content_length is not None and int(content_length) > settings.max_request_body_bytes:
                        return JSONResponse(status_code=413, content={"detail": "Request body too large"})
                except ValueError:
                    return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})

            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            response.headers["X-Request-ID"] = request_id_var.get()
            LOG.info(
                "request method=%s path=%s status=%s elapsed_ms=%.2f",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
            return response
        finally:
            request_id_var.reset(token)
