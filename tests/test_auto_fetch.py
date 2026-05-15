from __future__ import annotations

import logging
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# yfinance is a real dep in requirements.txt; we monkeypatch it for tests so
# nothing hits the network. The module is still imported lazily inside
# ``auto_fetch_outcome`` so we can swap it via ``sys.modules`` per test.


@dataclass(frozen=True)
class _Bar:
    Open: float | None
    High: float | None
    Low: float | None
    Close: float | None


class _StubDataFrame:
    """Minimal stand-in for the ``pandas.DataFrame`` returned by ``yfinance.Ticker.history``.

    Only the surface area used by :func:`auto_fetch_outcome` is implemented:
    ``empty``, ``sort_index()``, and ``itertuples()``.
    """

    def __init__(self, bars: list[_Bar]) -> None:
        self._bars = list(bars)

    @property
    def empty(self) -> bool:
        return not self._bars

    def sort_index(self) -> _StubDataFrame:
        return _StubDataFrame(list(self._bars))

    def itertuples(self) -> Iterator[_Bar]:
        return iter(self._bars)


class _StubTicker:
    def __init__(self, bars: list[_Bar]) -> None:
        self._bars = bars
        self.calls: list[dict[str, Any]] = []

    def history(self, *, start: str, end: str, auto_adjust: bool = False) -> _StubDataFrame:
        self.calls.append({"start": start, "end": end, "auto_adjust": auto_adjust})
        return _StubDataFrame(self._bars)


def _install_yfinance_stub(monkeypatch: pytest.MonkeyPatch, ticker: _StubTicker) -> None:
    mod = types.ModuleType("yfinance")

    def _ticker_factory(symbol: str) -> _StubTicker:
        return ticker

    mod.Ticker = _ticker_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", mod)


def test_auto_fetch_extracts_all_fields(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from equity_analyst.outcome_tracker import auto_fetch_outcome

    bars = [
        _Bar(Open=100.0, High=105.0, Low=98.0, Close=102.0),  # earnings day
        _Bar(Open=102.5, High=103.5, Low=100.5, Close=101.0),  # next trading day
        _Bar(Open=101.0, High=103.0, Low=100.0, Close=102.5),
        _Bar(Open=102.5, High=104.0, Low=101.5, Close=103.0),
        _Bar(Open=103.0, High=105.0, Low=102.0, Close=104.0),
        _Bar(Open=104.0, High=106.0, Low=103.0, Close=105.5),  # ~1 week later close
        _Bar(Open=105.5, High=107.0, Low=104.5, Close=106.0),
    ]
    ticker = _StubTicker(bars)
    _install_yfinance_stub(monkeypatch, ticker)

    caplog.set_level(logging.INFO, logger="equity_analyst.outcome_tracker")
    fetched = auto_fetch_outcome("CRCL", "Mon May 11 2026", baseline_close=99.0)

    assert fetched["earnings_day_open"] == pytest.approx(100.0)
    assert fetched["earnings_day_high"] == pytest.approx(105.0)
    assert fetched["earnings_day_low"] == pytest.approx(98.0)
    assert fetched["earnings_day_close"] == pytest.approx(102.0)
    assert fetched["next_trading_day_open"] == pytest.approx(102.5)
    assert fetched["next_trading_day_close"] == pytest.approx(101.0)
    assert fetched["one_week_later_close"] == pytest.approx(105.5)
    # 102 vs baseline 99 → +3.03% → "up".
    assert fetched["direction_vs_prior_close"] == "up"

    # The window passed to yfinance starts on the parsed earnings date and ends ~15 days later.
    assert ticker.calls[0]["start"] == "2026-05-11"
    assert ticker.calls[0]["end"] == "2026-05-26"


def test_auto_fetch_empty_dataframe_returns_all_none(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from equity_analyst.outcome_tracker import auto_fetch_outcome

    ticker = _StubTicker([])  # ADR / recent-IPO scenario
    _install_yfinance_stub(monkeypatch, ticker)

    caplog.set_level(logging.WARNING, logger="equity_analyst.outcome_tracker")
    fetched = auto_fetch_outcome("OBSCURE", "Mon May 11 2026", baseline_close=42.0)

    for key in (
        "earnings_day_open",
        "earnings_day_high",
        "earnings_day_low",
        "earnings_day_close",
        "next_trading_day_open",
        "next_trading_day_close",
        "one_week_later_close",
        "direction_vs_prior_close",
    ):
        assert fetched[key] is None
    assert any("empty bars" in rec.getMessage() for rec in caplog.records)


def test_auto_fetch_direction_down_and_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    from equity_analyst.outcome_tracker import auto_fetch_outcome

    bars_down = [_Bar(Open=100.0, High=101.0, Low=90.0, Close=92.0)]
    _install_yfinance_stub(monkeypatch, _StubTicker(bars_down))
    down = auto_fetch_outcome("X", "Tue May 12 2026", baseline_close=100.0)
    assert down["direction_vs_prior_close"] == "down"
    assert down["earnings_day_close"] == pytest.approx(92.0)
    # Without a 5th-trading-day bar, fall back to the only available close.
    assert down["one_week_later_close"] is None  # only one bar; len(bars) < 2 path
    assert down["next_trading_day_close"] is None

    bars_flat = [_Bar(Open=100.0, High=100.05, Low=99.95, Close=100.0)]
    _install_yfinance_stub(monkeypatch, _StubTicker(bars_flat))
    flat = auto_fetch_outcome("Y", "Tue May 12 2026", baseline_close=100.0)
    assert flat["direction_vs_prior_close"] == "flat"


def test_auto_fetch_partial_bars_falls_back_for_one_week(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from equity_analyst.outcome_tracker import auto_fetch_outcome

    bars = [
        _Bar(Open=10.0, High=11.0, Low=9.5, Close=10.5),
        _Bar(Open=10.5, High=11.2, Low=10.4, Close=10.8),
        _Bar(Open=10.8, High=11.0, Low=10.7, Close=10.9),
    ]
    _install_yfinance_stub(monkeypatch, _StubTicker(bars))

    caplog.set_level(logging.WARNING, logger="equity_analyst.outcome_tracker")
    fetched = auto_fetch_outcome("Z", "Mon May 11 2026")

    assert fetched["earnings_day_close"] == pytest.approx(10.5)
    assert fetched["next_trading_day_close"] == pytest.approx(10.8)
    # 3 bars total < 6 → falls back to last available close (10.9), with WARNING.
    assert fetched["one_week_later_close"] == pytest.approx(10.9)
    assert any("only 3 bars after earnings" in rec.getMessage() for rec in caplog.records)
    # direction stays None because baseline_close wasn't provided.
    assert fetched["direction_vs_prior_close"] is None


def test_auto_fetch_unparseable_date(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from equity_analyst.outcome_tracker import auto_fetch_outcome

    # Even with yfinance available, an unparseable date should short-circuit.
    _install_yfinance_stub(monkeypatch, _StubTicker([]))
    caplog.set_level(logging.WARNING, logger="equity_analyst.outcome_tracker")

    fetched = auto_fetch_outcome("X", "not-a-date-at-all", baseline_close=10.0)
    assert all(v is None for v in fetched.values())
    assert any("could not parse earnings_date" in rec.getMessage() for rec in caplog.records)


def test_auto_fetch_yfinance_exception_logs_and_returns_none(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from equity_analyst.outcome_tracker import auto_fetch_outcome

    class _BoomTicker:
        def history(self, **_kw: Any) -> Any:
            raise RuntimeError("yfinance went sideways")

    mod = types.ModuleType("yfinance")
    mod.Ticker = lambda _sym: _BoomTicker()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", mod)

    caplog.set_level(logging.WARNING, logger="equity_analyst.outcome_tracker")
    fetched = auto_fetch_outcome("X", "Mon May 11 2026")
    assert all(v is None for v in fetched.values())
    assert any("yfinance.history call failed" in rec.getMessage() for rec in caplog.records)


def test_auto_fetch_skips_future_earnings_calendar_date(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No Yahoo request when earnings session is still in the future (no bars yet)."""
    from equity_analyst import outcome_tracker as ot

    monkeypatch.setattr(ot, "_ny_calendar_date_today", lambda: date(2026, 5, 15))

    calls: list[tuple[Any, ...]] = []

    class _SpyTicker:
        def history(self, *args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            return pd.DataFrame()

    mod = types.ModuleType("yfinance")
    mod.Ticker = lambda _sym: _SpyTicker()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", mod)

    caplog.set_level(logging.INFO, logger="equity_analyst.outcome_tracker")
    fetched = ot.auto_fetch_outcome("NVDA", "Mon May 19 2026")
    assert calls == []
    assert all(v is None for v in fetched.values())
    assert any("skipping Yahoo fetch" in rec.getMessage() for rec in caplog.records)
    assert any("no historical bars yet" in rec.getMessage() for rec in caplog.records)


def test_resolve_baseline_close_prefers_config_then_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json as _json

    from equity_analyst.outcome_tracker import resolve_baseline_close_for_auto_fetch

    run_dir = tmp_path / "CRCL_20260511T023700Z"
    run_dir.mkdir()
    # Case 1: config has current_price → wins.
    (run_dir / "run.json").write_text(
        _json.dumps({"config": {"current_price": 113.45, "symbol": "CRCL", "earnings_date": "Mon May 11 2026"}}),
        encoding="utf-8",
    )
    (run_dir / "synthesis.md").write_text(
        "last verified close $999.00 from sources ...\n",
        encoding="utf-8",
    )
    assert resolve_baseline_close_for_auto_fetch(run_dir) == pytest.approx(113.45)

    # Case 2: no current_price → fall back to synthesis regex.
    (run_dir / "run.json").write_text(
        _json.dumps({"config": {"current_price": None, "symbol": "CRCL", "earnings_date": "Mon May 11 2026"}}),
        encoding="utf-8",
    )
    assert resolve_baseline_close_for_auto_fetch(run_dir) == pytest.approx(999.0)

    # Case 3: neither YAML nor synthesis hint → yfinance prior close (stub empty → None).
    (run_dir / "synthesis.md").write_text("no price phrase here", encoding="utf-8")

    class _EmptyPriorTicker:
        def history(self, **_kw: Any) -> pd.DataFrame:
            return pd.DataFrame()

    mod_empty = types.ModuleType("yfinance")
    mod_empty.Ticker = lambda _sym: _EmptyPriorTicker()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", mod_empty)
    assert resolve_baseline_close_for_auto_fetch(run_dir) is None


def test_fetch_prior_session_close_ignores_earnings_day_bar(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Last bar strictly before earnings_date wins (not same calendar day as earnings)."""
    from equity_analyst.outcome_tracker import fetch_prior_session_close_yfinance

    ny = "America/New_York"
    idx = pd.to_datetime(
        ["2026-05-06", "2026-05-07", "2026-05-08"],
    ).tz_localize(ny)
    frame = pd.DataFrame(
        {"Close": [70.0, 75.0, 78.42]},
        index=idx,
    )

    class _Ticker:
        def history(self, **_kw: Any) -> pd.DataFrame:
            return frame

    mod = types.ModuleType("yfinance")
    mod.Ticker = lambda _sym: _Ticker()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", mod)

    caplog.set_level(logging.INFO, logger="equity_analyst.outcome_tracker")
    out = fetch_prior_session_close_yfinance("CRCL", date(2026, 5, 11))
    assert out == pytest.approx(78.42)
    assert any(
        "baseline_close from prior session yfinance: 78.42 (date 2026-05-08)" in rec.getMessage()
        for rec in caplog.records
    )


def test_fetch_prior_session_close_monday_earnings_prior_is_friday(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from equity_analyst.outcome_tracker import fetch_prior_session_close_yfinance

    ny = "America/New_York"
    idx = pd.to_datetime(["2026-05-07", "2026-05-08"]).tz_localize(ny)
    frame = pd.DataFrame({"Close": [74.0, 80.0]}, index=idx)

    class _Ticker:
        def history(self, **_kw: Any) -> pd.DataFrame:
            return frame

    mod = types.ModuleType("yfinance")
    mod.Ticker = lambda _sym: _Ticker()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", mod)

    # Monday 2026-05-11: last session before is Friday 2026-05-08.
    assert fetch_prior_session_close_yfinance("X", date(2026, 5, 11)) == pytest.approx(80.0)


def test_resolve_baseline_explicit_current_price_skips_yfinance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import json as _json

    from equity_analyst.outcome_tracker import resolve_baseline_close_for_auto_fetch

    def _boom_ticker(_symbol: str) -> Any:
        raise AssertionError("yfinance.Ticker should not be used when baseline comes from run.json")

    mod = types.ModuleType("yfinance")
    mod.Ticker = _boom_ticker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", mod)

    run_dir = tmp_path / "RUN1"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        _json.dumps(
            {"config": {"current_price": 50.0, "symbol": "ZZZ", "earnings_date": "Mon May 11 2026"}},
        ),
        encoding="utf-8",
    )
    assert resolve_baseline_close_for_auto_fetch(run_dir) == pytest.approx(50.0)


def test_fetch_earnings_day_intraday_high_low_yfinance_validates_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from equity_analyst.outcome_tracker import fetch_earnings_day_intraday_high_low_yfinance

    def _bad(_symbol: str, _earnings_date: str) -> dict[str, Any]:
        return {
            "earnings_day_open": 1.0,
            "earnings_day_high": 3.0,
            "earnings_day_low": 5.0,
            "earnings_day_close": 2.0,
            "next_trading_day_open": None,
            "next_trading_day_close": None,
            "one_week_later_close": None,
            "direction_vs_prior_close": None,
        }

    monkeypatch.setattr(
        "equity_analyst.outcome_tracker.auto_fetch_outcome",
        _bad,
    )
    assert fetch_earnings_day_intraday_high_low_yfinance("X", "May 1, 2026") == (None, None)

    def _good(_symbol: str, _earnings_date: str) -> dict[str, Any]:
        return {
            "earnings_day_open": 1.0,
            "earnings_day_high": 110.0,
            "earnings_day_low": 100.0,
            "earnings_day_close": 105.0,
            "next_trading_day_open": None,
            "next_trading_day_close": None,
            "one_week_later_close": None,
            "direction_vs_prior_close": None,
        }

    monkeypatch.setattr(
        "equity_analyst.outcome_tracker.auto_fetch_outcome",
        _good,
    )
    assert fetch_earnings_day_intraday_high_low_yfinance("X", "May 1, 2026") == (100.0, 110.0)
