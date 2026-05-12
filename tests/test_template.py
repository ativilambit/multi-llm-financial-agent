from __future__ import annotations

from pathlib import Path

import pytest

from equity_analyst.config import RunConfig
from equity_analyst.prompting import render_prompt


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_template_renders_mndy_config_no_placeholders() -> None:
    cfg = RunConfig.model_validate(
        {
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
    assert "source URL and timestamp" in text
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
            "symbol": "ABCD",
            "today_date": "Tue Jan 1, 2026",
            "today_session": "regular hours",
            "earnings_date": "Wed Jan 15 2026",
            "target_dates": ["Wed Jan 15"],
            "next_trading_day": "Thu Jan 16",
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
    assert "Earnings date: Wed Jan 15 2026" in text
    assert "next trading day" in text
    assert "end of that earnings week" in text


def test_template_same_day_intraday_available_injects_bounds() -> None:
    cfg = RunConfig.model_validate(
        {
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
    assert "Verified options chain not available in this run" not in text


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
        }
    )
    text = render_prompt(cfg, _repo_root() / "prompts" / "equity_analyst.j2").text
    assert "Verified options chain not available in this run" in text
    assert "Yahoo Options" in text
