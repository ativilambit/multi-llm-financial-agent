"""Tests for session OHLC lock helpers (yfinance mocked; no live DB)."""

from __future__ import annotations

import types
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from equity_analyst.session_ohlc_lock import (
    fetch_session_daily_ohlc_yfinance,
    gha_auto_skip_reason,
    pick_session_bar_from_history,
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
