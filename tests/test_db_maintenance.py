from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_select_run_ids_missing_outcomes_queries_db(monkeypatch: pytest.MonkeyPatch) -> None:
    from equity_analyst import db_maintenance as dm

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = ["RUN_A", "RUN_B"]
    mock_session = MagicMock()
    mock_session.execute = AsyncMock(return_value=mock_result)

    @contextlib.asynccontextmanager
    async def _fake_session(*_a: object, **_kw: object):
        yield mock_session

    monkeypatch.setattr(dm, "get_async_session", _fake_session)
    monkeypatch.setattr(dm, "is_db_available", AsyncMock(return_value=True))

    out = await dm.select_run_ids_missing_outcomes(
        lookback_days=10,
        limit=5,
        symbols=["aa", "bb"],
        database_url=None,
    )
    assert out == ["RUN_A", "RUN_B"]
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_select_run_ids_missing_predictions_queries_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from equity_analyst import db_maintenance as dm

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = ["P1"]
    mock_session = MagicMock()
    mock_session.execute = AsyncMock(return_value=mock_result)

    @contextlib.asynccontextmanager
    async def _fake_session(*_a: object, **_kw: object):
        yield mock_session

    monkeypatch.setattr(dm, "get_async_session", _fake_session)
    monkeypatch.setattr(dm, "is_db_available", AsyncMock(return_value=True))

    out = await dm.select_run_ids_missing_predictions(
        lookback_days=3,
        limit=9,
        symbols=None,
        database_url=None,
    )
    assert out == ["P1"]
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_select_run_ids_missing_outcomes_requires_db(monkeypatch: pytest.MonkeyPatch) -> None:
    from equity_analyst import db_maintenance as dm

    monkeypatch.setattr(dm, "is_db_available", AsyncMock(return_value=False))
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        await dm.select_run_ids_missing_outcomes(lookback_days=1, limit=1, symbols=None)
