from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from sql_finance_copilot.api.dependencies import get_copilot_service
from sql_finance_copilot.api.schemas import SchemaRequest, SchemaResponse, SchemaTable
from sql_finance_copilot.api.services.copilot_service import CopilotService

router = APIRouter(tags=["schema"])


@router.post("/schema", response_model=SchemaResponse)
async def schema_endpoint(
    request: SchemaRequest,
    service: CopilotService = Depends(get_copilot_service),
) -> SchemaResponse:
    try:
        context = await run_in_threadpool(service.retrieve_schema, request.question, request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SchemaResponse(
        question=context.question,
        table_names=context.table_names,
        qualified_table_names=context.qualified_table_names,
        prompt_text=context.prompt_text,
        documents=[
            SchemaTable(
                doc_id=document.doc_id,
                schema_name=document.schema_name,
                table_name=document.table_name,
                text=document.text,
            )
            for document in context.documents
        ],
    )
