from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sql_finance_copilot.api.middleware import install_middleware
from sql_finance_copilot.api.routers import health_router, query_router, repair_router, schema_router
from sql_finance_copilot.config import AppSettings
from sql_finance_copilot.logging_config import configure_logging


def create_app() -> FastAPI:
    settings = AppSettings.from_env()
    configure_logging(settings.log_level)
    app = FastAPI(title="SQL Finance Copilot", version="0.1.0")
    app.state.settings = settings
    app.state.copilot = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    install_middleware(app, settings)
    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(repair_router)
    app.include_router(schema_router)

    return app


app = create_app()
