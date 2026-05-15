"""Tests for session OHLC lock helpers (yfinance mocked; no live DB)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import types
from datetime import UTC, date, datetime
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from equity_analyst.session_ohlc_lock import (
    SessionDailyOhlc,
    fetch_session_daily_ohlc_yfinance,
    gha_auto_skip_reason,
    pick_session_bar_from_history,
    run_lock_session_ohlc_cli,
)


def test_pick_session_bar_from_history_matches_ny_session_date() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2026-05-14 18:00:00", tz="UTC")])
    df = pd.DataFrame(
        {"Open": [10.0], "High": [12.0], "Low": [9.5], "Close": [11.0]},
        index=idx,
    )
    assert pick_session_bar_from_history(df, date(2026, 5, 13)) is None

    out = pick_session_bar_from_history(df, date(2026, 5, 14))
    assert out is not None
    assert out.open == pytest.approx(10.0)
    assert out.high == pytest.approx(12.0)
    assert out.low == pytest.approx(9.5)
    assert out.close == pytest.approx(11.0)


def test_gha_auto_skip_reason_weekday_before_close() -> None:
    # 2026-05-14 20:10 UTC = 16:10 America/New_York (EDT)
    assert gha_auto_skip_reason(now_utc=datetime(2026, 5, 14, 20, 10, tzinfo=UTC)) is not None


def test_gha_auto_skip_reason_weekday_after_close() -> None:
    # 2026-05-14 20:20 UTC = 16:20 America/New_York (EDT)
    assert gha_auto_skip_reason(now_utc=datetime(2026, 5, 14, 20, 20, tzinfo=UTC)) is None


def test_gha_auto_skip_reason_saturday() -> None:
    now_utc = datetime(2026, 5, 16, 20, 20, tzinfo=UTC)
    assert gha_auto_skip_reason(now_utc=now_utc) == "weekend America/New_York"


def test_fetch_session_daily_ohlc_uses_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    class _StubTicker:
        def history(self, *args: object, **kwargs: object) -> pd.DataFrame:
            idx = pd.DatetimeIndex([pd.Timestamp("2026-05-14", tz="America/New_York")])
            return pd.DataFrame(
                {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5]},
                index=idx,
            )

    yf_mod = types.ModuleType("yfinance")
    yf_mod.Ticker = lambda _s: _StubTicker()
    monkeypatch.setitem(sys.modules, "yfinance", yf_mod)

    out = fetch_session_daily_ohlc_yfinance("X", date(2026, 5, 14))
    assert out is not None
    assert out.close == pytest.approx(1.5)


def _gha_cli_ns(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "database_url": "postgresql://localhost/test",
        "dry_run": False,
        "gha_auto": True,
        "run_id": None,
        "symbols": None,
        "symbol": None,
        "symbols_env": False,
        "date": None,
        "session_partial": False,
        "lookback_days": 14,
        "runs_env": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_gha_auto_db_discovery_zero_symbols_exits_zero_info(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(
        "equity_analyst.session_ohlc_lock.is_db_available", AsyncMock(return_value=True)
    )
    monkeypatch.setattr("equity_analyst.session_ohlc_lock.gha_auto_skip_reason", lambda **_: None)
    discover = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "equity_analyst.session_ohlc_lock.discover_symbols_pending_session_ohlc_for_ny_day",
        discover,
    )
    rc = asyncio.run(run_lock_session_ohlc_cli(_gha_cli_ns()))
    assert rc == 0
    assert "no eligible runs" in caplog.text
    discover.assert_awaited_once()


def test_gha_auto_db_discovery_iterates_discovered_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "equity_analyst.session_ohlc_lock.is_db_available", AsyncMock(return_value=True)
    )
    monkeypatch.setattr("equity_analyst.session_ohlc_lock.gha_auto_skip_reason", lambda **_: None)
    discover = AsyncMock(return_value=["AA"])
    monkeypatch.setattr(
        "equity_analyst.session_ohlc_lock.discover_symbols_pending_session_ohlc_for_ny_day",
        discover,
    )
    monkeypatch.setattr(
        "equity_analyst.session_ohlc_lock.resolve_run_ids_for_symbol_session_day",
        AsyncMock(return_value=["run_1"]),
    )
    monkeypatch.setattr(
        "equity_analyst.session_ohlc_lock.load_run_symbols",
        AsyncMock(return_value={"run_1": "AA"}),
    )
    ohlc = SessionDailyOhlc(open=1.0, high=2.0, low=0.5, close=1.5)
    monkeypatch.setattr(
        "equity_analyst.session_ohlc_lock.fetch_session_daily_ohlc_yfinance", lambda *_a, **_k: ohlc
    )
    apply_mock = AsyncMock(return_value=1)
    monkeypatch.setattr(
        "equity_analyst.session_ohlc_lock.apply_session_ohlc_to_run_ids", apply_mock
    )
    rc = asyncio.run(run_lock_session_ohlc_cli(_gha_cli_ns(lookback_days=7)))
    assert rc == 0
    discover.assert_awaited_once()
    assert discover.await_args is not None
    assert discover.await_args.kwargs["lookback_days"] == 7
    assert apply_mock.await_count == 1
