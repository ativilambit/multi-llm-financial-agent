from __future__ import annotations

from pathlib import Path

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

    for i in range(1, 12):
        assert f"{i}." in text
    assert "12." not in text
    assert "13." not in text
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
    assert "~$105.5" in text
    assert "web_search" in text
    assert "Wed Feb 18 2026" in text
