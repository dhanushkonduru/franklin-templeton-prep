from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from sql_finance_copilot.api.dependencies import get_copilot_service
from sql_finance_copilot.api.schemas import RepairRequest, RepairResponse
from sql_finance_copilot.api.services.copilot_service import CopilotService

router = APIRouter(tags=["repair"])


@router.post("/repair", response_model=RepairResponse)
async def repair_endpoint(
    request: RepairRequest,
    service: CopilotService = Depends(get_copilot_service),
) -> RepairResponse:
    try:
        repaired_sql = await run_in_threadpool(
            service.repair_sql,
            request.question,
            request.sql,
            request.error,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return RepairResponse(question=request.question, original_sql=request.sql, repaired_sql=repaired_sql)
