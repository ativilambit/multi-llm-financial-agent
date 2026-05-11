from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def test_get_async_engine_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    import equity_analyst.db as db

    db._ENGINE = None
    db._SESSIONMAKER = None

    created: list[str] = []

    def _fake_create_async_engine(url: str, **_kw: object) -> object:
        created.append(url)
        return SimpleNamespace(url=url)

    monkeypatch.setattr(db, "create_async_engine", _fake_create_async_engine)

    e1 = db.get_async_engine(database_url="postgresql+psycopg://x:y@localhost:5432/z")
    e2 = db.get_async_engine(database_url="postgresql+psycopg://different:ignored@localhost:5432/ignored")

    assert e1 is e2
    assert created == ["postgresql+psycopg://x:y@localhost:5432/z"]


@pytest.mark.asyncio
async def test_is_db_available_returns_false_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import equity_analyst.db as db

    class _BadConn:
        async def __aenter__(self) -> _BadConn:
            raise RuntimeError("no connection")

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    class _BadEngine:
        def connect(self) -> _BadConn:
            return _BadConn()

    monkeypatch.setattr(db, "get_async_engine", lambda **_kw: _BadEngine())
    assert await db.is_db_available() is False


@pytest.mark.asyncio
async def test_best_effort_insert_logs_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    from equity_analyst.config import RunConfig
    from equity_analyst.db_ops import best_effort_upsert_run_and_responses

    cfg = RunConfig.model_validate(
        {
            "symbol": "CRCL",
            "today_date": "Mon May 11 2026",
            "today_session": "regular",
            "earnings_date": "Mon May 11 2026",
            "next_trading_day": "Tue May 12 2026",
            "followup_open_date": "Tue May 12 2026",
        }
    )

    monkeypatch.setattr("equity_analyst.db_ops.is_db_available", AsyncMock(return_value=True))

    @contextlib.asynccontextmanager
    async def _bad_session(*_a: object, **_kw: object):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    monkeypatch.setattr("equity_analyst.db_ops.get_async_session", _bad_session)

    caplog.set_level(logging.WARNING)
    monkeypatch.chdir(tmp_path)
    run_dir = Path("outputs") / "CRCL_20260511T123456Z"
    run_dir.mkdir(parents=True, exist_ok=True)
    await best_effort_upsert_run_and_responses(
        cfg=cfg,
        run_dir=run_dir,
        run_json_data={},
        started_at_utc=None,
        finished_at_utc=None,
        provider_responses=[],
        synthesis_path=run_dir / "synthesis.md",
        database_url=None,
    )

    assert any("DB insert failed" in r.message for r in caplog.records)

