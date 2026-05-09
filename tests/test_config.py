from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from equity_analyst.config import RunConfig, SynthesizerConfig, load_config
from equity_analyst.providers.gemini_provider import DEFAULT_GEMINI_MODEL


def test_providers_object_form_and_defaults() -> None:
    cfg = RunConfig.model_validate(
        {
            "symbol": "X",
            "today_low": 1,
            "today_high": 2,
            "current_price": 1.5,
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "earnings_timing": "t",
            "target_dates": [],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "providers": [
                {"name": "anthropic", "model": "claude-opus-4-7", "web_search": True},
                {"name": "openai", "request_timeout_s": 120},
            ],
        }
    )
    assert cfg.providers[0].name == "anthropic"
    assert cfg.providers[0].model == "claude-opus-4-7"
    assert cfg.providers[0].web_search is True
    assert cfg.providers[1].model is None
    assert cfg.providers[1].request_timeout_s == 120


def test_providers_legacy_string_list() -> None:
    cfg = RunConfig.model_validate(
        {
            "symbol": "X",
            "today_low": 1,
            "today_high": 2,
            "current_price": 1.5,
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "earnings_timing": "t",
            "target_dates": [],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "providers": ["openai", "gemini"],
        }
    )
    assert [p.name for p in cfg.providers] == ["openai", "gemini"]
    assert all(p.model is None for p in cfg.providers)


def test_synthesizer_string_and_object_form() -> None:
    c1 = RunConfig.model_validate(
        {
            "symbol": "X",
            "today_low": 1,
            "today_high": 2,
            "current_price": 1.5,
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "earnings_timing": "t",
            "target_dates": [],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "providers": ["openai"],
            "synthesizer": "gemini",
        }
    )
    assert c1.synthesizer == SynthesizerConfig(name="gemini")

    c2 = RunConfig.model_validate(
        {
            "symbol": "X",
            "today_low": 1,
            "today_high": 2,
            "current_price": 1.5,
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "earnings_timing": "t",
            "target_dates": [],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "providers": ["openai"],
            "synthesizer": {
                "name": "gemini",
                "model": "gemini-2.5-pro",
                "web_search": False,
                "request_timeout_s": 240,
            },
        }
    )
    assert c2.synthesizer.name == "gemini"
    assert c2.synthesizer.model == "gemini-2.5-pro"
    assert c2.synthesizer.web_search is False
    assert c2.synthesizer.request_timeout_s == 240


def test_default_synthesizer_max_output_tokens() -> None:
    cfg = RunConfig.model_validate(
        {
            "symbol": "X",
            "today_low": 1,
            "today_high": 2,
            "current_price": 1.5,
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "earnings_timing": "t",
            "target_dates": [],
            "next_trading_day": "n",
            "followup_open_date": "f",
        }
    )
    assert cfg.synthesizer_max_output_tokens == 24_000
    assert cfg.max_output_tokens == 16_000
    assert cfg.synthesizer_max_output_tokens != cfg.max_output_tokens
    assert cfg.request_timeout_s == 180.0


def test_provider_config_optional_max_output_tokens() -> None:
    cfg = RunConfig.model_validate(
        {
            "symbol": "X",
            "today_low": 1,
            "today_high": 2,
            "current_price": 1.5,
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "earnings_timing": "t",
            "target_dates": [],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "providers": [
                {"name": "anthropic", "max_output_tokens": 24_000},
                {"name": "grok", "max_output_tokens": 12_000},
                {"name": "openai"},
            ],
        }
    )
    assert cfg.providers[0].max_output_tokens == 24_000
    assert cfg.providers[1].max_output_tokens == 12_000
    assert cfg.providers[2].max_output_tokens is None


def test_prompt_cache_enabled_defaults_true() -> None:
    cfg = RunConfig.model_validate(
        {
            "symbol": "X",
            "today_low": 1,
            "today_high": 2,
            "current_price": 1.5,
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "earnings_timing": "t",
            "target_dates": [],
            "next_trading_day": "n",
            "followup_open_date": "f",
        }
    )
    assert cfg.prompt_cache_enabled is True


def test_default_synthesizer_is_gemini() -> None:
    cfg = RunConfig.model_validate(
        {
            "symbol": "X",
            "today_low": 1,
            "today_high": 2,
            "current_price": 1.5,
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "earnings_timing": "t",
            "target_dates": [],
            "next_trading_day": "n",
            "followup_open_date": "f",
        }
    )
    assert cfg.synthesizer.name == "gemini"


@pytest.mark.parametrize(
    ("providers", "synth", "msg"),
    [
        ([{"name": "unknown"}], None, "Unknown provider"),
        (["openai"], "not-a-provider", "Unknown synthesizer"),
    ],
)
def test_unknown_provider_or_synthesizer_rejected(
    providers: Any, synth: Any, msg: str
) -> None:
    base: dict[str, Any] = {
        "symbol": "X",
        "today_low": 1,
        "today_high": 2,
        "current_price": 1.5,
        "today_date": "d",
        "today_session": "s",
        "earnings_date": "e",
        "earnings_timing": "t",
        "target_dates": [],
        "next_trading_day": "n",
        "followup_open_date": "f",
        "providers": providers,
    }
    if synth is not None:
        base["synthesizer"] = synth
    with pytest.raises(ValueError, match=msg):
        RunConfig.model_validate(base)


def test_mndy_fast_config_hybrid_web_search_and_shared_fields_match_standard() -> None:
    repo = Path(__file__).resolve().parents[1]
    standard_path = repo / "configs" / "mndy_2026_05_08.yaml"
    fast_path = repo / "configs" / "mndy_2026_05_08_fast.yaml"

    standard_raw = yaml.safe_load(standard_path.read_text(encoding="utf-8"))
    fast_raw = yaml.safe_load(fast_path.read_text(encoding="utf-8"))

    def _without_providers_synth(d: dict[str, Any]) -> dict[str, Any]:
        out = dict(d)
        out.pop("providers", None)
        out.pop("synthesizer", None)
        return out

    assert _without_providers_synth(standard_raw) == _without_providers_synth(fast_raw)

    fast_cfg = load_config(str(fast_path))
    by_name = {p.name: p for p in fast_cfg.providers}
    assert by_name["anthropic"].web_search is False
    assert by_name["grok"].web_search is False
    assert by_name["openai"].web_search is True
    assert fast_cfg.synthesizer.web_search is False


def test_mndy_configs_use_latest_gemini_pro_synthesizer() -> None:
    repo = Path(__file__).resolve().parents[1]
    for filename in ("mndy_2026_05_08.yaml", "mndy_2026_05_08_fast.yaml"):
        cfg = load_config(str(repo / "configs" / filename))
        assert cfg.synthesizer.name == "gemini"
        assert cfg.synthesizer.model == DEFAULT_GEMINI_MODEL
