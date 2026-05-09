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
            "earnings_timing": "early morning et, before the market open",
            "target_dates": ["Mon May 11", "Fri May 15", "Fri May 22", "Fri May 29", "Fri Jun 5"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last two months back", "last month", "last week", "yesterday", "today"],
            "providers": ["anthropic", "openai"],
            "synthesizer": "anthropic",
        }
    )

    prompt_path = _repo_root() / "prompts" / "equity_analyst.j2"
    rendered = render_prompt(cfg, prompt_path)
    text = rendered.text

    for i in range(1, 14):
        assert f"{i}." in text

    assert "{{" not in text
    assert "MNDY" in text
    assert "$68" in text
    assert "$74" in text
    assert "$73.24" in text
    assert "Fri May 8, 2026" in text
    assert "Mon May 11 2026" in text


def test_template_generalizes_to_other_symbol() -> None:
    cfg = RunConfig.model_validate(
        {
            "symbol": "NVDA",
            "company_name": "NVIDIA",
            "today_low": 100,
            "today_high": 110,
            "current_price": 105.5,
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
    assert "$100" in text
    assert "$110" in text
    assert "$105.5" in text
    assert "Wed Feb 18 2026" in text

