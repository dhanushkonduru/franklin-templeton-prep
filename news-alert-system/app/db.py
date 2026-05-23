from __future__ import annotations

from collections.abc import AsyncIterator
from urllib.parse import urlparse

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.models import Base


def create_engine_and_sessionmaker(database_url: str | None = None) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    resolved_url = database_url or settings.postgres_dsn
    parsed = urlparse(resolved_url.replace("+asyncpg", "").replace("+aiosqlite", ""))
    engine_kwargs: dict[str, object] = {"pool_pre_ping": True, "future": True}
    if parsed.scheme.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    else:
        engine_kwargs["pool_size"] = 5
        engine_kwargs["max_overflow"] = 10

    engine = create_async_engine(resolved_url, **engine_kwargs)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    return engine, sessionmaker


engine, async_session_factory = create_engine_and_sessionmaker()


async def init_db(engine_override: AsyncEngine | None = None) -> None:
    active_engine = engine_override or engine
    async with active_engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with async_session_factory() as session:
        yield session
