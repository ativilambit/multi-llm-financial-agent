from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from equity_analyst.config import (
    RunConfig,
    env_from_persisted_run_json,
    run_profile_from_persisted_run_json,
)
from equity_analyst.db_ops import best_effort_upsert_run_and_responses


def test_env_from_persisted_run_json_top_level_wins() -> None:
    d = {"env": "test", "config": {"env": "production"}}
    assert env_from_persisted_run_json(d) == "test"


def test_env_from_persisted_run_json_config_fallback() -> None:
    d = {"config": {"env": "test"}}
    assert env_from_persisted_run_json(d) == "test"


def test_env_from_persisted_run_json_legacy_defaults_production() -> None:
    d = {"config": {"symbol": "X"}}
    assert env_from_persisted_run_json(d) == "production"
def test_run_profile_from_persisted_run_json_top_level_wins() -> None:
    d = {"run_profile": "dev", "config": {"run_profile": "production"}}
    assert run_profile_from_persisted_run_json(d) == "dev"


def test_run_profile_from_persisted_run_json_config_fallback() -> None:
    d = {"config": {"run_profile": "production"}}
    assert run_profile_from_persisted_run_json(d) == "production"


def test_run_profile_from_persisted_run_json_legacy_defaults_production() -> None:
    d = {"config": {"symbol": "X"}}
    assert run_profile_from_persisted_run_json(d) == "production"


@pytest.mark.asyncio
async def test_best_effort_upsert_skips_when_dev_profile_and_production_env(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    async def _should_not_run(*_a: object, **_kw: object) -> bool:
        raise AssertionError("is_db_available should not run when persistence is disabled")

    monkeypatch.setattr("equity_analyst.db_ops.is_db_available", _should_not_run)
    caplog.set_level(logging.INFO)
    monkeypatch.chdir(tmp_path)
    run_dir = Path("outputs") / "X_20260511T000000Z"
    run_dir.mkdir(parents=True)
    cfg = RunConfig.model_validate(
        {
            "symbol": "X",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "next_trading_day": "n",
            "followup_open_date": "f",
            "run_profile": "dev",
            "env": "production",
        }
    )
    await best_effort_upsert_run_and_responses(
        cfg=cfg,
        run_dir=run_dir,
        run_json_data={"dry_run": False},
        started_at_utc=None,
        finished_at_utc=None,
        provider_responses=[],
        synthesis_path=run_dir / "synthesis.md",
        database_url=None,
    )
    assert any(
        "DB write skipped: run_profile=dev env=production (need production profile or test tier)"
        in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_best_effort_upsert_calls_db_when_test_env_and_dev_profile(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    import contextlib

    monkeypatch.setattr("equity_analyst.db_ops.is_db_available", AsyncMock(return_value=True))

    @contextlib.asynccontextmanager
    async def _bad_session(*_a: object, **_kw: object):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    monkeypatch.setattr("equity_analyst.db_ops.get_async_session", _bad_session)

    caplog.set_level(logging.WARNING)
    monkeypatch.chdir(tmp_path)
    run_dir = Path("outputs") / "X_20260511T000000Z"
    run_dir.mkdir(parents=True)
    cfg = RunConfig.model_validate(
        {
            "symbol": "X",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "next_trading_day": "n",
            "followup_open_date": "f",
            "run_profile": "dev",
            "env": "test",
        }
    )
    await best_effort_upsert_run_and_responses(
        cfg=cfg,
        run_dir=run_dir,
        run_json_data={"dry_run": False},
        started_at_utc=None,
        finished_at_utc=None,
        provider_responses=[],
        synthesis_path=run_dir / "synthesis.md",
        database_url=None,
    )
    assert any("DB insert failed" in r.message for r in caplog.records)
@pytest.mark.asyncio
async def test_best_effort_upsert_calls_db_path_when_production(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    import contextlib

    monkeypatch.setattr("equity_analyst.db_ops.is_db_available", AsyncMock(return_value=True))

    @contextlib.asynccontextmanager
    async def _bad_session(*_a: object, **_kw: object):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    monkeypatch.setattr("equity_analyst.db_ops.get_async_session", _bad_session)

    caplog.set_level(logging.WARNING)
    monkeypatch.chdir(tmp_path)
    run_dir = Path("outputs") / "X_20260511T000000Z"
    run_dir.mkdir(parents=True)
    cfg = RunConfig.model_validate(
        {
            "symbol": "X",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "next_trading_day": "n",
            "followup_open_date": "f",
            "run_profile": "production",
        }
    )
    await best_effort_upsert_run_and_responses(
        cfg=cfg,
        run_dir=run_dir,
        run_json_data={"dry_run": False},
        started_at_utc=None,
        finished_at_utc=None,
        provider_responses=[],
        synthesis_path=run_dir / "synthesis.md",
        database_url=None,
    )
    assert any("DB insert failed" in r.message for r in caplog.records)
