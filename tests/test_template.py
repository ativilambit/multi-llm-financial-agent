from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from equity_analyst.config import RunConfig
from equity_analyst.prompting import render_prompt

_TEMPLATE_CFG_NET_OFF: dict[str, Any] = {
    "options_chain_auto_fetch": False,
    "run_profile": "production",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_template_renders_mndy_config_no_placeholders() -> None:
    cfg = RunConfig.model_validate(
        {
            **_TEMPLATE_CFG_NET_OFF,
            "symbol": "MNDY",
            "company_name": None,
            "today_low": 68,
            "today_high": 74,
            "current_price": 73.24,
            "today_date": "Fri May 8, 2026",
            "today_session": "after the market trading window",
            "earnings_date": "Mon May 11 2026",
            "target_dates": ["Mon May 11", "Fri May 15", "Fri May 22", "Fri May 29", "Fri Jun 5"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 6,
            "short_interest_lookbacks": ["last two months back", "last month", "last week", "yesterday", "today"],
            "providers": ["anthropic", "openai"],
            "synthesizer": "anthropic",
        }
    )

    prompt_path = _repo_root() / "prompts" / "equity_analyst.j2"
    rendered = render_prompt(cfg, prompt_path)
    text = rendered.text

    for i in range(1, 13):
        assert f"{i}." in text
    assert "13." not in text
    assert "14." not in text
    assert "Evaluate other prompts" not in text
    assert "Suggest additional prompts" not in text
    assert "\n5. Also, what were all the relevant short interest" in text

    assert "{{" not in text
    assert "MNDY" in text
    assert "web_search" in text
    assert "last regular-session" in text or "last verified close" in text
    assert "unverified" in text
    assert "reference" in text.lower()
    assert "Fri May 8, 2026" in text
    assert "Mon May 11 2026" in text
    assert "Earnings call timing (mandatory verification):" in text
    assert "NOT provided in this brief" in text
    assert "Yahoo Finance Earnings Calendar" in text
    assert "Date anchors" in text
    assert "Earnings date: Mon May 11 2026" in text
    assert "Next trading day after earnings: Tues May 12" in text
    assert "Target dates (open/close anchors): Mon May 11, Fri May 15, Fri May 22, Fri May 29, Fri Jun 5" in text
    assert "Follow-up open date (~1 week after earnings): Mon May 18" in text
    assert "on the day of the earnings call" in text
    assert "next trading day" in text
    assert "end of that earnings week" in text
    assert "one trading week after" in text
    assert "Bottom-up qualitative overlay" in text
    assert "directional bias" in text
    assert "### Qualitative evidence" in text
    assert "Source:" in text and ("http://" in text or "https://" in text)
    assert "sections 1, 9, and 11" in text or "(sections 1, 9, 11)" in text

    sigma = "\N{GREEK SMALL LETTER SIGMA}"
    ndash = "\N{EN DASH}"
    assert f"1{sigma}" in text
    assert f"2{sigma}" in text
    assert f"3{sigma}" in text
    assert f"  - 1{sigma}: $X.XX {ndash} $X.XX (±Y.Y%)" in text
    assert f"  - 2{sigma}: $X.XX {ndash} $X.XX (±Y.Y%)" in text
    assert f"  - 3{sigma}: $X.XX {ndash} $X.XX (±Y.Y%)" in text

    assert "same_day_intraday_available" in text
    assert "`same_day_intraday_available`=False" in text
    assert "fall back" in text.lower()
    assert "prior-close anchored" in text
    assert "SD / range anchoring rule" in text


def test_template_uses_config_earnings_timing_when_provided() -> None:
    cfg = RunConfig.model_validate(
        {
            **_TEMPLATE_CFG_NET_OFF,
            "symbol": "ZZZ",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "earnings_timing": "after market close (legacy hint)",
            "target_dates": ["t1"],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "historical_quarters": 4,
            "short_interest_lookbacks": ["last week"],
            "providers": ["openai"],
            "synthesizer": "openai",
        }
    )
    text = render_prompt(cfg, _repo_root() / "prompts" / "equity_analyst.j2").text
    assert "{{" not in text
    assert "Earnings timing (from this brief): after market close (legacy hint)" in text
    assert "Earnings call timing (mandatory verification):" not in text
    assert "Date anchors" in text
    assert "Earnings date: e" in text
    assert "day of the earnings call" in text


def test_template_renders_when_reference_prices_omitted() -> None:
    cfg = RunConfig.model_validate(
        {
            **_TEMPLATE_CFG_NET_OFF,
            "symbol": "ABCD",
            "today_date": "Tue Jan 1, 2026",
            "today_session": "regular hours",
            "earnings_date": "Thu Jan 15 2026",
            "target_dates": ["Thu Jan 15"],
            "next_trading_day": "Fri Jan 16",
            "followup_open_date": "Mon Jan 20",
            "historical_quarters": 4,
            "short_interest_lookbacks": ["last week"],
            "providers": ["openai"],
            "synthesizer": "openai",
        }
    )
    prompt_path = _repo_root() / "prompts" / "equity_analyst.j2"
    text = render_prompt(cfg, prompt_path).text
    assert "{{" not in text
    assert "User session labels (not prices):" in text
    assert "ABCD" in text
    assert "Date anchors" in text
    assert "Earnings date: Thu Jan 15 2026" in text
    assert "next trading day" in text
    assert "end of that earnings week" in text


def test_template_same_day_intraday_available_injects_bounds() -> None:
    cfg = RunConfig.model_validate(
        {
            **_TEMPLATE_CFG_NET_OFF,
            "symbol": "TEST",
            "today_date": "Tue May 12, 2026",
            "today_session": "regular hours",
            "earnings_date": "Mon May 11, 2026",
            "target_dates": ["Mon May 11"],
            "next_trading_day": "Tue May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 4,
            "short_interest_lookbacks": ["last week"],
            "providers": ["openai"],
            "synthesizer": "openai",
            "same_day_intraday_min": 100.0,
            "same_day_intraday_max": 110.0,
        }
    )
    text = render_prompt(cfg, _repo_root() / "prompts" / "equity_analyst.j2").text
    assert "`same_day_intraday_available`=True" in text
    assert "`same_day_intraday_min`=100.0" in text
    assert "`same_day_intraday_max`=110.0" in text
    assert "[99.00, 111.00]" in text


def test_render_prompt_auto_fetch_fills_intraday_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_fetch(symbol: str, earnings_date: str) -> tuple[float | None, float | None]:
        assert symbol == "X"
        assert "2026" in earnings_date
        return 50.0, 52.5

    monkeypatch.setattr(
        "equity_analyst.outcome_tracker.fetch_earnings_day_intraday_high_low_yfinance",
        _fake_fetch,
    )
    cfg = RunConfig.model_validate(
        {
            **_TEMPLATE_CFG_NET_OFF,
            "symbol": "X",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "May 12, 2026",
            "target_dates": ["t1"],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "historical_quarters": 4,
            "short_interest_lookbacks": ["last week"],
            "providers": ["openai"],
            "synthesizer": "openai",
            "same_day_intraday_auto_fetch": True,
        }
    )
    rendered = render_prompt(cfg, _repo_root() / "prompts" / "equity_analyst.j2")
    assert rendered.context["same_day_intraday_available"] is True
    assert rendered.context["same_day_intraday_min"] == 50.0
    assert rendered.context["same_day_intraday_max"] == 52.5
    assert rendered.context["same_day_intraday_anchor_band_low"] == 49.0
    assert rendered.context["same_day_intraday_anchor_band_high"] == 53.5
    assert "`same_day_intraday_available`=True" in rendered.text


def test_template_generalizes_to_other_symbol() -> None:
    cfg = RunConfig.model_validate(
        {
            **_TEMPLATE_CFG_NET_OFF,
            "symbol": "NVDA",
            "company_name": "NVIDIA",
            "reference_session_low": 100,
            "reference_session_high": 110,
            "reference_last_price": 105.5,
            "today_date": "Mon Jan 5, 2026",
            "today_session": "during regular trading hours",
            "earnings_date": "Wed Feb 18 2026",
            "earnings_timing": "after market close",
            "target_dates": ["Wed Feb 18", "Fri Feb 20", "Fri Feb 27", "Fri Mar 6", "Fri Mar 13"],
            "next_trading_day": "Thu Feb 19",
            "followup_open_date": "Mon Feb 23",
            "historical_quarters": 8,
            "short_interest_lookbacks": ["last month", "last week", "today"],
            "providers": ["anthropic"],
            "synthesizer": "anthropic",
        }
    )

    prompt_path = _repo_root() / "prompts" / "equity_analyst.j2"
    rendered = render_prompt(cfg, prompt_path)
    text = rendered.text

    assert "{{" not in text
    assert "NVDA" in text
    assert "$100.0\u2013$110.0" in text


def test_template_options_chain_markdown_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    md = "| test table |\n|--|\n| x |"

    def _fake_resolve(cfg: RunConfig) -> tuple[dict, str]:
        assert cfg.symbol == "ACME"
        return (
            {
                "options_chain_available": True,
                "symbol": "ACME",
                "spot": 10.0,
                "available_expiries": ["2026-06-01"],
                "selected_expiries": [],
                "as_of": "z",
                "fetch_error": None,
            },
            md,
        )

    monkeypatch.setattr("equity_analyst.prompting._resolve_options_chain", _fake_resolve)
    cfg = RunConfig.model_validate(
        {
            **_TEMPLATE_CFG_NET_OFF,
            "symbol": "ACME",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "target_dates": ["t1"],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "historical_quarters": 4,
            "short_interest_lookbacks": ["last week"],
            "providers": ["openai"],
            "synthesizer": "openai",
        }
    )
    text = render_prompt(cfg, _repo_root() / "prompts" / "equity_analyst.j2").text
    assert "**Verified options chain (use these numbers, do not fabricate):**" in text
    assert "| test table |" in text
    assert "Verified options chain not fetched" not in text
    assert "Verified options chain fetch failed" not in text


def test_template_options_chain_fallback_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_resolve(cfg: RunConfig) -> tuple[dict, str]:
        return (
            {
                "options_chain_available": False,
                "symbol": cfg.symbol,
                "spot": None,
                "available_expiries": [],
                "selected_expiries": [],
                "as_of": "z",
                "fetch_error": None,
            },
            "",
        )

    monkeypatch.setattr("equity_analyst.prompting._resolve_options_chain", _fake_resolve)
    cfg = RunConfig.model_validate(
        {
            **_TEMPLATE_CFG_NET_OFF,
            "symbol": "ZZZ",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "target_dates": ["t1"],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "historical_quarters": 4,
            "short_interest_lookbacks": ["last week"],
            "providers": ["openai"],
            "synthesizer": "openai",
            "options_chain_auto_fetch": True,
        }
    )
    text = render_prompt(cfg, _repo_root() / "prompts" / "equity_analyst.j2").text
    assert "Verified options chain has no listed expiries" in text
    assert "Yahoo Options" in text


def test_template_options_chain_fallback_when_auto_fetch_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_resolve(cfg: RunConfig) -> tuple[dict, str]:
        return (
            {
                "options_chain_available": False,
                "symbol": cfg.symbol,
                "spot": None,
                "available_expiries": [],
                "selected_expiries": [],
                "as_of": "z",
                "fetch_error": None,
            },
            "",
        )

    monkeypatch.setattr("equity_analyst.prompting._resolve_options_chain", _fake_resolve)
    cfg = RunConfig.model_validate(
        {
            **_TEMPLATE_CFG_NET_OFF,
            "symbol": "OFF",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "target_dates": ["t1"],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "historical_quarters": 4,
            "short_interest_lookbacks": ["last week"],
            "providers": ["openai"],
            "synthesizer": "openai",
            "options_chain_auto_fetch": False,
        }
    )
    text = render_prompt(cfg, _repo_root() / "prompts" / "equity_analyst.j2").text
    assert "Verified options chain not fetched (auto-fetch disabled)" in text


def test_template_options_chain_fallback_includes_fetch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_resolve(cfg: RunConfig) -> tuple[dict, str]:
        return (
            {
                "options_chain_available": False,
                "symbol": cfg.symbol,
                "spot": None,
                "available_expiries": [],
                "selected_expiries": [],
                "as_of": "z",
                "fetch_error": "unit-test synthetic failure",
            },
            "",
        )

    monkeypatch.setattr("equity_analyst.prompting._resolve_options_chain", _fake_resolve)
    cfg = RunConfig.model_validate(
        {
            **_TEMPLATE_CFG_NET_OFF,
            "symbol": "BAD",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "target_dates": ["t1"],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "historical_quarters": 4,
            "short_interest_lookbacks": ["last week"],
            "providers": ["openai"],
            "synthesizer": "openai",
            "options_chain_auto_fetch": True,
        }
    )
    text = render_prompt(cfg, _repo_root() / "prompts" / "equity_analyst.j2").text
    assert "Verified options chain fetch failed (unit-test synthetic failure)" in text


def test_render_prompt_warns_when_options_chain_auto_fetch_but_unavailable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _fake_resolve(cfg: RunConfig) -> tuple[dict, str]:
        return (
            {
                "options_chain_available": False,
                "symbol": cfg.symbol,
                "spot": None,
                "available_expiries": [],
                "selected_expiries": [],
                "as_of": "z",
                "fetch_error": "unit-test log reason",
            },
            "",
        )

    monkeypatch.setattr("equity_analyst.prompting._resolve_options_chain", _fake_resolve)
    cfg = RunConfig.model_validate(
        {
            **_TEMPLATE_CFG_NET_OFF,
            "symbol": "LOG",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "target_dates": ["t1"],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "historical_quarters": 4,
            "short_interest_lookbacks": ["last week"],
            "providers": ["openai"],
            "synthesizer": "openai",
            "options_chain_auto_fetch": True,
        }
    )
    with caplog.at_level(logging.WARNING, logger="equity_analyst.prompting"):
        render_prompt(cfg, _repo_root() / "prompts" / "equity_analyst.j2")
    assert "unit-test log reason" in caplog.text


def test_template_includes_iv_crush_multiplier_and_adjusted_daily_vol_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_resolve(cfg: RunConfig) -> tuple[dict, str]:
        return (
            {
                "options_chain_available": True,
                "symbol": cfg.symbol,
                "spot": 100.0,
                "available_expiries": ["2026-05-15", "2026-05-22"],
                "selected_expiries": [
                    {
                        "expiry_date": "2026-05-15",
                        "dte": 2,
                        "atm_strike": 100.0,
                        "atm_call_iv": 1.03,
                        "atm_put_iv": 1.03,
                        "atm_straddle_mid": 11.0,
                        "implied_move_pct": 0.11,
                        "expected_move_dollar": 11.0,
                        "skew_25d_call_minus_put_iv": 0.0,
                        "skew_25d_note": "n/a",
                        "total_call_volume": 1,
                        "total_put_volume": 1,
                        "put_call_ratio": 1.0,
                        "total_call_oi": 1,
                        "total_put_oi": 1,
                        "put_call_ratio_oi": 1.0,
                        "atm_call_bid": 1.0,
                        "atm_call_ask": 1.0,
                        "atm_call_mid": 1.0,
                        "atm_call_last": 1.0,
                        "atm_put_bid": 1.0,
                        "atm_put_ask": 1.0,
                        "atm_put_mid": 1.0,
                        "atm_put_last": 1.0,
                    },
                    {
                        "expiry_date": "2026-05-22",
                        "dte": 7,
                        "atm_strike": 100.0,
                        "atm_call_iv": 0.61,
                        "atm_put_iv": 0.61,
                        "atm_straddle_mid": 8.0,
                        "implied_move_pct": 0.08,
                        "expected_move_dollar": 8.0,
                        "skew_25d_call_minus_put_iv": 0.0,
                        "skew_25d_note": "n/a",
                        "total_call_volume": 1,
                        "total_put_volume": 1,
                        "put_call_ratio": 1.0,
                        "total_call_oi": 1,
                        "total_put_oi": 1,
                        "put_call_ratio_oi": 1.0,
                        "atm_call_bid": 1.0,
                        "atm_call_ask": 1.0,
                        "atm_call_mid": 1.0,
                        "atm_call_last": 1.0,
                        "atm_put_bid": 1.0,
                        "atm_put_ask": 1.0,
                        "atm_put_mid": 1.0,
                        "atm_put_last": 1.0,
                    },
                ],
                "as_of": "z",
                "fetch_error": None,
            },
            "|stub chain md|\n",
        )

    monkeypatch.setattr("equity_analyst.prompting._resolve_options_chain", _fake_resolve)
    monkeypatch.setattr("equity_analyst.prompting.fetch_hv30_annualized_percent", lambda _sym: 84.9)
    cfg = RunConfig.model_validate(
        {
            **_TEMPLATE_CFG_NET_OFF,
            "symbol": "NBIS",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "2026-05-13",
            "target_dates": ["t1"],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "historical_quarters": 4,
            "short_interest_lookbacks": ["last week"],
            "providers": ["openai"],
            "synthesizer": "openai",
        }
    )
    rendered = render_prompt(cfg, _repo_root() / "prompts" / "equity_analyst.j2")
    text = rendered.text
    assert "Pre-computed IV crush" in text
    assert "iv_crush_multiplier" in text
    mult = 0.61 / 1.03
    assert f"{mult:.4f}" in text
    assert rendered.context["iv_crush_multiplier"] == pytest.approx(mult)
    assert rendered.context["hv30_annualized_pct"] == pytest.approx(84.9)
    assert rendered.context["daily_vol_iv_adjusted"] == pytest.approx((84.9 / (252**0.5)) * mult)
    assert "IV-crush adjustment" in text


def test_template_falls_back_when_multiplier_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_resolve(cfg: RunConfig) -> tuple[dict, str]:
        return (
            {
                "options_chain_available": True,
                "symbol": cfg.symbol,
                "spot": 100.0,
                "available_expiries": ["2026-05-15"],
                "selected_expiries": [
                    {
                        "expiry_date": "2026-05-15",
                        "dte": 2,
                        "atm_strike": 100.0,
                        "atm_call_iv": 1.03,
                        "atm_put_iv": 1.03,
                        "atm_straddle_mid": 11.0,
                        "implied_move_pct": 0.11,
                        "expected_move_dollar": 11.0,
                        "skew_25d_call_minus_put_iv": 0.0,
                        "skew_25d_note": "n/a",
                        "total_call_volume": 1,
                        "total_put_volume": 1,
                        "put_call_ratio": 1.0,
                        "total_call_oi": 1,
                        "total_put_oi": 1,
                        "put_call_ratio_oi": 1.0,
                        "atm_call_bid": 1.0,
                        "atm_call_ask": 1.0,
                        "atm_call_mid": 1.0,
                        "atm_call_last": 1.0,
                        "atm_put_bid": 1.0,
                        "atm_put_ask": 1.0,
                        "atm_put_mid": 1.0,
                        "atm_put_last": 1.0,
                    },
                ],
                "as_of": "z",
                "fetch_error": None,
            },
            "|stub|",
        )

    monkeypatch.setattr("equity_analyst.prompting._resolve_options_chain", _fake_resolve)
    monkeypatch.setattr("equity_analyst.prompting.fetch_hv30_annualized_percent", lambda _sym: 84.9)
    cfg = RunConfig.model_validate(
        {
            **_TEMPLATE_CFG_NET_OFF,
            "symbol": "NBIS",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "2026-05-13",
            "target_dates": ["t1"],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "historical_quarters": 4,
            "short_interest_lookbacks": ["last week"],
            "providers": ["openai"],
            "synthesizer": "openai",
        }
    )
    rendered = render_prompt(cfg, _repo_root() / "prompts" / "equity_analyst.j2")
    assert rendered.context["iv_crush_multiplier"] is None
    assert rendered.context["daily_vol_iv_adjusted"] is None
    assert "Pre-computed IV crush" not in rendered.text
