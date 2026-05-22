from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from fastapi import Request

from sql_finance_copilot.api.services.copilot_service import CopilotService
from sql_finance_copilot.config import AppSettings
from sql_finance_copilot.core.orchestrator import SqlCopilot


def get_settings(request: Request) -> AppSettings:
    return request.app.state.settings


def get_copilot(request: Request) -> SqlCopilot:
    cached = getattr(request.app.state, "copilot", None)
    if cached is None:
        cached = SqlCopilot.build(request.app.state.settings)
        request.app.state.copilot = cached
    return cached


def get_copilot_service(copilot: Annotated[SqlCopilot, Depends(get_copilot)]) -> CopilotService:
    return CopilotService(copilot)
