"""Postgres-backed selection helpers for scheduled maintenance (outcomes / predictions)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.sql import and_

from equity_analyst.db import get_async_session, is_db_available
from equity_analyst.db_models import OutcomeRow, PredictionRow, RunRow


def _normalize_symbols(symbols: list[str] | None) -> list[str] | None:
    if not symbols:
        return None
    out = [s.strip().upper() for s in symbols if s and str(s).strip()]
    return out or None


async def select_run_ids_missing_outcomes(
    *,
    lookback_days: int = 14,
    limit: int = 50,
    symbols: list[str] | None = None,
    database_url: str | None = None,
) -> list[str]:
    """Runs with no ``outcomes`` row, ``run_document`` present, and recent activity."""
    if not await is_db_available(database_url=database_url):
        raise RuntimeError("DATABASE_URL is unset or Postgres is unreachable")
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    if limit < 1:
        raise ValueError("limit must be >= 1")

    cutoff = datetime.now(tz=UTC) - timedelta(days=int(lookback_days))
    ts = func.coalesce(RunRow.finished_at_utc, RunRow.started_at_utc, RunRow.created_at_utc)

    conds: list[Any] = [
        RunRow.run_document.is_not(None),
        OutcomeRow.run_id.is_(None),
        ts >= cutoff,
    ]
    sym = _normalize_symbols(symbols)
    if sym is not None:
        conds.append(func.upper(RunRow.symbol).in_(sym))

    stmt = (
        select(RunRow.run_id)
        .select_from(RunRow)
        .outerjoin(OutcomeRow, OutcomeRow.run_id == RunRow.run_id)
        .where(and_(*conds))
        .order_by(ts.desc().nullslast())
        .limit(int(limit))
    )

    async with get_async_session(database_url=database_url) as session:
        res = await session.execute(stmt)
        return [str(r) for r in res.scalars().all()]


async def select_run_ids_missing_predictions(
    *,
    lookback_days: int = 14,
    limit: int = 50,
    symbols: list[str] | None = None,
    database_url: str | None = None,
) -> list[str]:
    """Runs with no ``predictions`` rows, ``run_document`` present, stored synthesis, recent activity."""
    if not await is_db_available(database_url=database_url):
        raise RuntimeError("DATABASE_URL is unset or Postgres is unreachable")
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    if limit < 1:
        raise ValueError("limit must be >= 1")

    cutoff = datetime.now(tz=UTC) - timedelta(days=int(lookback_days))
    ts = func.coalesce(RunRow.finished_at_utc, RunRow.started_at_utc, RunRow.created_at_utc)

    pred_exists = (
        select(PredictionRow.id).where(PredictionRow.run_id == RunRow.run_id).limit(1).exists()
    )

    conds: list[Any] = [
        RunRow.run_document.is_not(None),
        RunRow.synthesis_markdown.is_not(None),
        func.length(func.trim(RunRow.synthesis_markdown)) > 0,
        ~pred_exists,
        ts >= cutoff,
    ]
    sym = _normalize_symbols(symbols)
    if sym is not None:
        conds.append(func.upper(RunRow.symbol).in_(sym))

    stmt = (
        select(RunRow.run_id).where(and_(*conds)).order_by(ts.desc().nullslast()).limit(int(limit))
    )

    async with get_async_session(database_url=database_url) as session:
        res = await session.execute(stmt)
        return [str(r) for r in res.scalars().all()]
