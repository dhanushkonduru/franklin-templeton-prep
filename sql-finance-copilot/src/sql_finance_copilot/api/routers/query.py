from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from sql_finance_copilot.api.dependencies import get_copilot_service
from sql_finance_copilot.api.schemas import QueryRequest, QueryResponse
from sql_finance_copilot.api.services.copilot_service import CopilotService

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
@router.post("/v1/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    service: CopilotService = Depends(get_copilot_service),
) -> QueryResponse:
    try:
        artifact = await run_in_threadpool(service.run_query, request.question, request.chart)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return QueryResponse(
        question=artifact.question,
        sql=artifact.sql,
        validated_sql=artifact.validated_sql,
        rows=artifact.rows,
        columns=artifact.columns,
        row_count=artifact.row_count,
        truncated=artifact.truncated,
        schema_tables=artifact.schema_context.qualified_table_names,
        repaired=artifact.repaired,
        repair_attempts=artifact.repair_attempts,
        repair_log=artifact.repair_log,
        error=artifact.error,
        executed_at=artifact.executed_at,
        chart_type=artifact.chart.chart_type if artifact.chart else None,
        chart_x=artifact.chart.x if artifact.chart else None,
        chart_y=artifact.chart.y if artifact.chart else None,
        chart_title=artifact.chart.title if artifact.chart else None,
    )
