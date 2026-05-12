from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from equity_analyst.options_chain import (
    ExpirySnapshot,
    OptionsChainSnapshot,
    _select_relevant_expiries,
    clear_options_chain_cache,
    fetch_options_chain_snapshot,
    options_chain_expiry_audit_messages,
    options_chain_snapshot_from_prompt_dict,
)


def test_select_relevant_expiries_may_earnings_week() -> None:
    avail = [
        date(2026, 5, 8),
        date(2026, 5, 15),
        date(2026, 5, 22),
        date(2026, 5, 29),
        date(2026, 6, 19),
    ]
    earn = date(2026, 5, 13)
    picked = _select_relevant_expiries(avail, earn)
    assert date(2026, 5, 15) in picked
    assert date(2026, 5, 22) in picked  # closest listed expiry to earnings + ~7d after earnings


def test_to_prompt_dict_shape_and_straddle_math() -> None:
    ex = ExpirySnapshot(
        expiry_date="2026-05-15",
        dte=4,
        atm_strike=100.0,
        atm_call_bid=5.0,
        atm_call_ask=7.0,
        atm_call_mid=6.0,
        atm_call_last=6.0,
        atm_call_iv=0.5,
        atm_put_bid=4.0,
        atm_put_ask=6.0,
        atm_put_mid=5.0,
        atm_put_last=5.0,
        atm_put_iv=0.48,
        atm_straddle_mid=11.0,
        implied_move_pct=0.11,
        expected_move_dollar=11.0,
        skew_25d_call_minus_put_iv=0.02,
        skew_25d_note="test",
        total_call_volume=1000,
        total_put_volume=2000,
        put_call_ratio=2.0,
        total_call_oi=5000,
        total_put_oi=2500,
        put_call_ratio_oi=0.5,
    )
    snap = OptionsChainSnapshot(
        as_of="2026-05-12T12:00:00Z",
        symbol="TEST",
        spot=100.0,
        available_expiries=["2026-05-15", "2026-05-22"],
        selected_expiries=[ex],
    )
    d = snap.to_prompt_dict()
    assert d["options_chain_available"] is True
    assert d["symbol"] == "TEST"
    assert d["spot"] == 100.0
    assert d["available_expiries"] == ["2026-05-15", "2026-05-22"]
    assert len(d["selected_expiries"]) == 1
    row = d["selected_expiries"][0]
    assert row["atm_straddle_mid"] == 11.0
    assert row["put_call_ratio"] == pytest.approx(2.0)
    assert row["implied_move_pct"] == pytest.approx(0.11)
    md = snap.to_markdown_table()
    assert "Verified options chain" in md
    assert "2026-05-15" in md
    assert "11.0" in md or "11" in md


def test_options_chain_snapshot_from_prompt_dict_roundtrip() -> None:
    snap = OptionsChainSnapshot(
        as_of="z",
        symbol="ZZ",
        spot=50.0,
        available_expiries=["2026-01-17"],
        selected_expiries=[],
    )
    back = options_chain_snapshot_from_prompt_dict(snap.to_prompt_dict())
    assert back is not None
    assert back.symbol == "ZZ"
    assert back.available_expiries == ["2026-01-17"]


def test_fetch_options_chain_snapshot_mocked_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_options_chain_cache()

    calls = pd.DataFrame(
        {
            "strike": [99.0, 100.0, 101.0],
            "bid": [4.0, 5.0, 3.0],
            "ask": [6.0, 7.0, 4.0],
            "lastPrice": [5.0, 6.0, 3.5],
            "impliedVolatility": [0.4, 0.5, 0.45],
            "volume": [10, 20, 5],
            "openInterest": [100, 200, 50],
        }
    )
    puts = pd.DataFrame(
        {
            "strike": [99.0, 100.0, 101.0],
            "bid": [3.0, 4.0, 5.0],
            "ask": [5.0, 6.0, 7.0],
            "lastPrice": [4.0, 5.0, 6.0],
            "impliedVolatility": [0.42, 0.48, 0.52],
            "volume": [8, 15, 4],
            "openInterest": [90, 180, 40],
        }
    )

    class _Chain:
        def __init__(self) -> None:
            self.calls = calls
            self.puts = puts

    class _Ticker:
        options = ("2026-05-15", "2026-05-22")

        def __init__(self, symbol: str) -> None:
            self._sym = symbol

        def option_chain(self, expiry: str) -> _Chain:
            return _Chain()

        def history(self, *args: object, **kwargs: object) -> pd.DataFrame:
            return pd.DataFrame({"Close": [100.0]}, index=[pd.Timestamp("2026-05-12")])

    def _fake_ticker(symbol: str) -> _Ticker:
        return _Ticker(symbol)

    import equity_analyst.options_chain as oc

    monkeypatch.setattr(oc, "_parse_earnings_calendar_date", lambda s: date(2026, 5, 13))
    monkeypatch.setattr(oc, "_parse_today_date", lambda s: date(2026, 5, 12))
    monkeypatch.setattr(oc, "_resolve_spot", lambda _t: 100.0)

    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", _fake_ticker)

    snap = fetch_options_chain_snapshot(
        "DT",
        "ignored",
        ["Wed May 13"],
        today_date="Tue May 12, 2026",
    )
    assert snap is not None
    assert snap.symbol == "DT"
    assert snap.spot == 100.0
    assert "2026-05-15" in snap.available_expiries
    assert snap.selected_expiries
    atm = snap.selected_expiries[0]
    assert atm.atm_strike == 100.0
    assert atm.atm_straddle_mid == pytest.approx(11.0)
    assert atm.put_call_ratio == pytest.approx(27 / 35)
    d = snap.to_prompt_dict()
    assert d["options_chain_available"] is True


def test_options_chain_expiry_audit_messages_flags_unknown_expiry() -> None:
    sg = chr(0x03C3)
    syn = (
        "Wednesday, May 13\n"
        f"  - 3{sg}: $1 - $9 (+/-60%) derived from 2026-05-20 weekly expiry\n"
    )
    ver: dict = {
        "sigma_band_sessions": [
            {"session": "May 13", "sigma_baseline": "2026-05-20 weekly", "sigma_scaling_check_passed": True},
        ],
    }
    oc_data = {"options_chain_available": True, "available_expiries": ["2026-05-15", "2026-05-22"]}
    msgs = options_chain_expiry_audit_messages(syn, ver, options_chain_data=oc_data, symbol="DT")
    assert msgs
    assert "2026-05-20" in msgs[0]
    assert "verified chain" in msgs[0].lower()
