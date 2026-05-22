from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import create_engine, text

from sql_finance_copilot.api.schemas import HealthCheckResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_endpoint(request: Request) -> HealthCheckResponse:
    checks = {"app": "ok", "database": "unknown", "copilot": "not_initialized"}

    def _db_check() -> str:
        engine = create_engine(request.app.state.settings.database_url, future=True)
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return "ok"
        except Exception:
            return "error"
        finally:
            engine.dispose()

    checks["database"] = await run_in_threadpool(_db_check)
    if request.app.state.copilot is not None:
        checks["copilot"] = "initialized"

    status = "ok" if checks.get("database") == "ok" else "degraded"
    return HealthCheckResponse(status=status, checks=checks)
