from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _resolve_database_url(*, database_url: str | None = None) -> str | None:
    if database_url is not None and str(database_url).strip():
        return str(database_url).strip()
    raw = os.environ.get("DATABASE_URL")
    if raw is None or not str(raw).strip():
        return None
    return str(raw).strip()


def get_async_engine(*, database_url: str | None = None) -> AsyncEngine:
    """Get a process-wide async engine singleton.

    Uses ``DATABASE_URL`` env var by default, or a non-empty explicit override.
    """
    global _engine, _sessionmaker
    if _engine is not None:
        return _engine

    url = _resolve_database_url(database_url=database_url)
    if not url:
        raise RuntimeError("DATABASE_URL is not set")

    _engine = create_async_engine(url, pool_pre_ping=True)
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


@asynccontextmanager
async def get_async_session(*, database_url: str | None = None) -> AsyncIterator[AsyncSession]:
    """Context manager yielding an AsyncSession bound to the singleton engine."""
    global _sessionmaker
    if _sessionmaker is None:
        get_async_engine(database_url=database_url)
    assert _sessionmaker is not None
    async with _sessionmaker() as session:
        yield session


async def is_db_available(*, database_url: str | None = None) -> bool:
    try:
        engine = get_async_engine(database_url=database_url)
    except Exception:
        return False
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def _reset_db_state_for_tests() -> None:
    """Testing helper: clear engine/sessionmaker singletons."""
    global _engine, _sessionmaker
    _engine = None
    _sessionmaker = None

