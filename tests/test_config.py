from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from equity_analyst.config import RunConfig, SynthesizerConfig, load_config
from equity_analyst.providers.gemini_provider import DEFAULT_GEMINI_MODEL


def test_iterative_cost_optimization_flags_default_on() -> None:
    cfg = RunConfig.model_validate(
        {
            "symbol": "X",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "target_dates": [],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "providers": ["openai"],
        }
    )
    assert cfg.facts_packet_enabled is True
    assert cfg.conditional_fanout_enabled is True
    assert cfg.fan_out_on_continue is True
    assert cfg.refinement_mode_prompt_enabled is True
    assert cfg.options_chain_auto_fetch is True


def test_options_chain_auto_fetch_env_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIONS_CHAIN_AUTO_FETCH", "0")
    cfg = RunConfig.model_validate(
        {
            "symbol": "X",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "target_dates": [],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "providers": ["openai"],
        }
    )
    assert cfg.options_chain_auto_fetch is False
    monkeypatch.delenv("OPTIONS_CHAIN_AUTO_FETCH", raising=False)


def test_options_chain_auto_fetch_yaml_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIONS_CHAIN_AUTO_FETCH", "0")
    cfg = RunConfig.model_validate(
        {
            "symbol": "X",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "target_dates": [],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "providers": ["openai"],
            "options_chain_auto_fetch": True,
        }
    )
    assert cfg.options_chain_auto_fetch is True
    monkeypatch.delenv("OPTIONS_CHAIN_AUTO_FETCH", raising=False)


def test_options_chain_auto_fetch_invalid_env_warns_and_keeps_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("OPTIONS_CHAIN_AUTO_FETCH", "maybe")
    with caplog.at_level(logging.WARNING, logger="equity_analyst.config"):
        cfg = RunConfig.model_validate(
            {
                "symbol": "X",
                "today_date": "d",
                "today_session": "s",
                "earnings_date": "e",
                "target_dates": [],
                "next_trading_day": "n",
                "followup_open_date": "f",
                "providers": ["openai"],
            }
        )
    assert cfg.options_chain_auto_fetch is True
    assert "Invalid OPTIONS_CHAIN_AUTO_FETCH" in caplog.text
    monkeypatch.delenv("OPTIONS_CHAIN_AUTO_FETCH", raising=False)


def test_reference_price_yaml_aliases() -> None:
    cfg = RunConfig.model_validate(
        {
            "symbol": "X",
            "reference_session_low": 10.0,
            "reference_session_high": 12.0,
            "reference_last_price": 11.0,
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "earnings_timing": "t",
            "target_dates": [],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "providers": ["openai"],
        }
    )
    assert cfg.today_low == 10.0
    assert cfg.today_high == 12.0
    assert cfg.current_price == 11.0


def test_optional_price_hints_may_be_omitted() -> None:
    cfg = RunConfig.model_validate(
        {
            "symbol": "X",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "earnings_timing": "t",
            "target_dates": [],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "providers": ["openai"],
        }
    )
    assert cfg.today_low is None
    assert cfg.today_high is None
    assert cfg.current_price is None


def test_same_day_intraday_min_max_must_be_paired() -> None:
    base: dict[str, Any] = {
        "symbol": "X",
        "today_date": "d",
        "today_session": "s",
        "earnings_date": "e",
        "target_dates": [],
        "next_trading_day": "n",
        "followup_open_date": "f",
        "providers": ["openai"],
    }
    with pytest.raises(ValueError, match="same_day_intraday_min"):
        RunConfig.model_validate({**base, "same_day_intraday_min": 1.0})
    with pytest.raises(ValueError, match="same_day_intraday_min"):
        RunConfig.model_validate({**base, "same_day_intraday_max": 2.0})


def test_earnings_timing_may_be_omitted() -> None:
    cfg = RunConfig.model_validate(
        {
            "symbol": "X",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "target_dates": [],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "providers": ["openai"],
        }
    )
    assert cfg.earnings_timing is None


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
    assert cfg.verifier_max_output_tokens == 16_384
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


def test_oversized_summarize_defaults() -> None:
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
    assert cfg.summarize_oversized_providers is True
    assert cfg.summarize_threshold_input_tokens == 8000
    assert cfg.synthesizer_max_input_tokens == 100_000
    assert cfg.oversized_summarize_provider == "gemini"
    assert cfg.oversized_summarize_model == "gemini-3-flash-preview"
    assert cfg.oversized_summarize_max_output_tokens == 8192
    assert cfg.oversized_summarize_max_input_tokens == 100_000
    assert cfg.oversized_summarize_min_retention == 0.40
    assert cfg.oversized_summarize_fallback_provider is None


def test_oversized_summarize_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OVERSIZED_SUMMARIZE_PROVIDER", "openai")
    monkeypatch.setenv("OVERSIZED_SUMMARIZE_MODEL", "gpt-4o-mini")
    cfg = RunConfig.model_validate(_minimal_run_config_dict())
    assert cfg.oversized_summarize_provider == "openai"
    assert cfg.oversized_summarize_model == "gpt-4o-mini"
    monkeypatch.delenv("OVERSIZED_SUMMARIZE_PROVIDER", raising=False)
    monkeypatch.delenv("OVERSIZED_SUMMARIZE_MODEL", raising=False)


def test_oversized_summarize_yaml_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OVERSIZED_SUMMARIZE_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OVERSIZED_SUMMARIZE_PROVIDER", "openai")
    d = _minimal_run_config_dict()
    d["oversized_summarize_model"] = "gemini-3-flash-preview"
    d["oversized_summarize_provider"] = "gemini"
    cfg = RunConfig.model_validate(d)
    assert cfg.oversized_summarize_model == "gemini-3-flash-preview"
    assert cfg.oversized_summarize_provider == "gemini"
    monkeypatch.delenv("OVERSIZED_SUMMARIZE_MODEL", raising=False)
    monkeypatch.delenv("OVERSIZED_SUMMARIZE_PROVIDER", raising=False)


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


def test_default_verifier_provider_is_gemini() -> None:
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
    assert cfg.verifier_provider == "gemini"
    assert cfg.verifier_model is None


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


def test_crcl_fast_config_hybrid_web_search_and_shared_fields_match_standard() -> None:
    repo = Path(__file__).resolve().parents[1]
    standard_path = repo / "configs" / "crcl_2026_05_08.yaml"
    fast_path = repo / "configs" / "crcl_2026_05_08_fast.yaml"

    standard_raw = yaml.safe_load(standard_path.read_text(encoding="utf-8"))
    fast_raw = yaml.safe_load(fast_path.read_text(encoding="utf-8"))

    def _without_providers_synth(d: dict[str, Any]) -> dict[str, Any]:
        out = dict(d)
        out.pop("providers", None)
        out.pop("synthesizer", None)
        return out

    assert _without_providers_synth(standard_raw) == _without_providers_synth(fast_raw)

    def _strip_timing_fields(obj: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in obj.items() if k not in ("web_search", "request_timeout_s")}

    std_providers = standard_raw.get("providers") or []
    fast_providers = fast_raw.get("providers") or []
    assert len(std_providers) == len(fast_providers)
    for s, f in zip(std_providers, fast_providers, strict=True):
        assert _strip_timing_fields(s) == _strip_timing_fields(f)

    std_synth = standard_raw.get("synthesizer") or {}
    fast_synth = fast_raw.get("synthesizer") or {}
    assert isinstance(std_synth, dict) and isinstance(fast_synth, dict)
    assert _strip_timing_fields(std_synth) == _strip_timing_fields(fast_synth)

    fast_cfg = load_config(str(fast_path))
    by_name = {p.name: p for p in fast_cfg.providers}
    assert by_name["anthropic"].web_search is False
    assert by_name["grok"].web_search is False
    assert by_name["openai"].web_search is True
    assert by_name["gemini"].web_search is False
    assert fast_cfg.synthesizer.web_search is False


def test_mndy_configs_use_latest_gemini_pro_synthesizer() -> None:
    repo = Path(__file__).resolve().parents[1]
    for filename in ("mndy_2026_05_08.yaml", "mndy_2026_05_08_fast.yaml"):
        cfg = load_config(str(repo / "configs" / filename))
        assert cfg.synthesizer.name == "gemini"
        assert cfg.synthesizer.model == DEFAULT_GEMINI_MODEL
        assert cfg.verifier_provider == "gemini"
        assert cfg.verifier_model == DEFAULT_GEMINI_MODEL


def test_crcl_configs_use_latest_gemini_pro_synthesizer() -> None:
    repo = Path(__file__).resolve().parents[1]
    for filename in ("crcl_2026_05_08.yaml", "crcl_2026_05_08_fast.yaml"):
        cfg = load_config(str(repo / "configs" / filename))
        assert cfg.synthesizer.name == "gemini"
        assert cfg.synthesizer.model == DEFAULT_GEMINI_MODEL
        assert cfg.verifier_provider == "gemini"
        assert cfg.verifier_model == DEFAULT_GEMINI_MODEL


GEMINI_FAN_OUT_FLASH_MODEL = "gemini-3-flash-preview"


def test_mndy_standard_config_has_four_fan_out_providers_with_gemini_flash() -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(str(repo / "configs" / "mndy_2026_05_08.yaml"))
    names = [p.name for p in cfg.providers]
    assert names == ["anthropic", "openai", "grok", "gemini"]
    by_name = {p.name: p for p in cfg.providers}
    assert by_name["gemini"].model == GEMINI_FAN_OUT_FLASH_MODEL
    assert by_name["gemini"].request_timeout_s == 600


def test_mndy_fast_config_has_four_fan_out_providers_with_gemini_flash() -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(str(repo / "configs" / "mndy_2026_05_08_fast.yaml"))
    names = [p.name for p in cfg.providers]
    assert names == ["anthropic", "openai", "grok", "gemini"]
    by_name = {p.name: p for p in cfg.providers}
    assert by_name["gemini"].model == GEMINI_FAN_OUT_FLASH_MODEL
    assert by_name["gemini"].web_search is False
    assert by_name["gemini"].request_timeout_s == 180


def test_run_environment_yaml_and_default() -> None:
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
            "providers": ["openai"],
        }
    )
    assert cfg.run_environment == "production"

    cfg_test = RunConfig.model_validate(
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
            "run_environment": "test",
        }
    )
    assert cfg_test.run_environment == "test"


def test_run_environment_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUN_ENVIRONMENT", "test")
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
            "providers": ["openai"],
            "run_environment": "production",
        }
    )
    assert cfg.run_environment == "test"
    monkeypatch.delenv("RUN_ENVIRONMENT", raising=False)


def test_drive_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRIVE_UPLOAD_ENABLED", "true")
    monkeypatch.setenv("DRIVE_CREDENTIALS_PATH", "/tmp/sa.json")
    monkeypatch.setenv("DRIVE_ROOT_FOLDER_ID", "folder123")
    monkeypatch.setenv("DRIVE_AUTH_MODE", "oauth_user")
    monkeypatch.setenv("DRIVE_OAUTH_TOKEN_PATH", "/tmp/oauth-token.json")
    monkeypatch.setenv("DRIVE_OAUTH_CLIENT_SECRETS_PATH", "/tmp/oauth-client.json")
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
            "providers": ["openai"],
            "drive_upload_enabled": False,
            "drive_credentials_path": None,
            "drive_root_folder_id": None,
            "drive_auth_mode": "service_account",
            "drive_oauth_token_path": None,
            "drive_oauth_client_secrets_path": None,
        }
    )
    assert cfg.drive_upload_enabled is True
    assert cfg.drive_credentials_path == "/tmp/sa.json"
    assert cfg.drive_root_folder_id == "folder123"
    assert cfg.drive_auth_mode == "oauth_user"
    assert cfg.drive_oauth_token_path == "/tmp/oauth-token.json"
    assert cfg.drive_oauth_client_secrets_path == "/tmp/oauth-client.json"
    monkeypatch.delenv("DRIVE_UPLOAD_ENABLED", raising=False)
    monkeypatch.delenv("DRIVE_CREDENTIALS_PATH", raising=False)
    monkeypatch.delenv("DRIVE_ROOT_FOLDER_ID", raising=False)
    monkeypatch.delenv("DRIVE_AUTH_MODE", raising=False)
    monkeypatch.delenv("DRIVE_OAUTH_TOKEN_PATH", raising=False)
    monkeypatch.delenv("DRIVE_OAUTH_CLIENT_SECRETS_PATH", raising=False)


def test_pdf_output_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PDF_OUTPUT_ENABLED", "0")
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
            "providers": ["openai"],
            "pdf_output_enabled": True,
        }
    )
    assert cfg.pdf_output_enabled is False
    monkeypatch.delenv("PDF_OUTPUT_ENABLED", raising=False)


def _minimal_run_config_dict() -> dict[str, Any]:
    return {
        "symbol": "X",
        "today_date": "d",
        "today_session": "s",
        "earnings_date": "e",
        "target_dates": [],
        "next_trading_day": "n",
        "followup_open_date": "f",
        "providers": ["openai"],
    }


def test_sigma_variance_check_quorum_for_error_default_two() -> None:
    cfg = RunConfig.model_validate(_minimal_run_config_dict())
    assert cfg.sigma_variance_check_quorum_for_error == 2


def test_sigma_variance_check_quorum_for_error_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGMA_VARIANCE_CHECK_QUORUM_FOR_ERROR", "1")
    cfg = RunConfig.model_validate(_minimal_run_config_dict())
    assert cfg.sigma_variance_check_quorum_for_error == 1
    monkeypatch.delenv("SIGMA_VARIANCE_CHECK_QUORUM_FOR_ERROR", raising=False)


def test_sigma_variance_check_quorum_for_error_yaml_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGMA_VARIANCE_CHECK_QUORUM_FOR_ERROR", "1")
    d = _minimal_run_config_dict()
    d["sigma_variance_check_quorum_for_error"] = 4
    cfg = RunConfig.model_validate(d)
    assert cfg.sigma_variance_check_quorum_for_error == 4
    monkeypatch.delenv("SIGMA_VARIANCE_CHECK_QUORUM_FOR_ERROR", raising=False)


def test_retry_max_attempts_fan_out_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRY_MAX_ATTEMPTS_FAN_OUT", "7")
    cfg = RunConfig.model_validate(_minimal_run_config_dict())
    assert cfg.retry_max_attempts_fan_out == 7
    monkeypatch.delenv("RETRY_MAX_ATTEMPTS_FAN_OUT", raising=False)


def test_retry_max_attempts_fan_out_yaml_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRY_MAX_ATTEMPTS_FAN_OUT", "7")
    d = _minimal_run_config_dict()
    d["retry_max_attempts_fan_out"] = 4
    cfg = RunConfig.model_validate(d)
    assert cfg.retry_max_attempts_fan_out == 4
    monkeypatch.delenv("RETRY_MAX_ATTEMPTS_FAN_OUT", raising=False)


def test_final_report_full_synthesis_defaults_true() -> None:
    cfg = RunConfig.model_validate(_minimal_run_config_dict())
    assert cfg.final_report_full_synthesis is True


def test_final_report_full_synthesis_env_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINAL_REPORT_FULL_SYNTHESIS", "0")
    cfg = RunConfig.model_validate(_minimal_run_config_dict())
    assert cfg.final_report_full_synthesis is False
    monkeypatch.delenv("FINAL_REPORT_FULL_SYNTHESIS", raising=False)


def test_final_report_full_synthesis_yaml_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINAL_REPORT_FULL_SYNTHESIS", "0")
    d = _minimal_run_config_dict()
    d["final_report_full_synthesis"] = True
    cfg = RunConfig.model_validate(d)
    assert cfg.final_report_full_synthesis is True
    monkeypatch.delenv("FINAL_REPORT_FULL_SYNTHESIS", raising=False)


def test_facts_packet_max_output_tokens_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACTS_PACKET_MAX_OUTPUT_TOKENS", "4096")
    cfg = RunConfig.model_validate(_minimal_run_config_dict())
    assert cfg.facts_packet_max_output_tokens == 4096
    monkeypatch.delenv("FACTS_PACKET_MAX_OUTPUT_TOKENS", raising=False)


def test_facts_packet_max_output_tokens_yaml_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACTS_PACKET_MAX_OUTPUT_TOKENS", "8192")
    d = _minimal_run_config_dict()
    d["facts_packet_max_output_tokens"] = 512
    cfg = RunConfig.model_validate(d)
    assert cfg.facts_packet_max_output_tokens == 512
    monkeypatch.delenv("FACTS_PACKET_MAX_OUTPUT_TOKENS", raising=False)


def test_verifier_max_output_tokens_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERIFIER_MAX_OUTPUT_TOKENS", "32768")
    cfg = RunConfig.model_validate(_minimal_run_config_dict())
    assert cfg.verifier_max_output_tokens == 32_768
    monkeypatch.delenv("VERIFIER_MAX_OUTPUT_TOKENS", raising=False)


def test_verifier_max_output_tokens_yaml_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERIFIER_MAX_OUTPUT_TOKENS", "8192")
    d = _minimal_run_config_dict()
    d["verifier_max_output_tokens"] = 2048
    cfg = RunConfig.model_validate(d)
    assert cfg.verifier_max_output_tokens == 2048
    monkeypatch.delenv("VERIFIER_MAX_OUTPUT_TOKENS", raising=False)


@pytest.mark.parametrize("bad", ("abc", "0", "255", "1000000"))
def test_verifier_max_output_tokens_invalid_env_warns_and_keeps_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, bad: str
) -> None:
    monkeypatch.setenv("VERIFIER_MAX_OUTPUT_TOKENS", bad)
    with caplog.at_level(logging.WARNING, logger="equity_analyst.config"):
        cfg = RunConfig.model_validate(_minimal_run_config_dict())
    assert cfg.verifier_max_output_tokens == 16_384
    assert "Invalid VERIFIER_MAX_OUTPUT_TOKENS" in caplog.text
    monkeypatch.delenv("VERIFIER_MAX_OUTPUT_TOKENS", raising=False)


@pytest.mark.parametrize("bad", ("abc", "0", "255", "1000000"))
def test_facts_packet_max_output_tokens_invalid_env_warns_and_keeps_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, bad: str
) -> None:
    monkeypatch.setenv("FACTS_PACKET_MAX_OUTPUT_TOKENS", bad)
    with caplog.at_level(logging.WARNING, logger="equity_analyst.config"):
        cfg = RunConfig.model_validate(_minimal_run_config_dict())
    assert cfg.facts_packet_max_output_tokens == 4096
    assert "Invalid FACTS_PACKET_MAX_OUTPUT_TOKENS" in caplog.text
    monkeypatch.delenv("FACTS_PACKET_MAX_OUTPUT_TOKENS", raising=False)


def test_facts_packet_enabled_yaml_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACTS_PACKET_ENABLED", "true")
    d = _minimal_run_config_dict()
    d["facts_packet_enabled"] = False
    cfg = RunConfig.model_validate(d)
    assert cfg.facts_packet_enabled is False
    monkeypatch.delenv("FACTS_PACKET_ENABLED", raising=False)


def test_facts_packet_enabled_env_when_omitted_from_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACTS_PACKET_ENABLED", "0")
    cfg = RunConfig.model_validate(_minimal_run_config_dict())
    assert cfg.facts_packet_enabled is False
    monkeypatch.delenv("FACTS_PACKET_ENABLED", raising=False)


def test_drive_settings_loaded_from_dotenv_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DRIVE_UPLOAD_ENABLED", raising=False)
    monkeypatch.delenv("DRIVE_CREDENTIALS_PATH", raising=False)
    monkeypatch.delenv("DRIVE_ROOT_FOLDER_ID", raising=False)

    (tmp_path / ".env").write_text(
        "DRIVE_UPLOAD_ENABLED=true\n"
        "DRIVE_CREDENTIALS_PATH=/tmp/from-dotenv-sa.json\n"
        "DRIVE_ROOT_FOLDER_ID=folder-from-dotenv\n",
        encoding="utf-8",
    )
    cfg_yaml = tmp_path / "minimal.yaml"
    cfg_yaml.write_text(
        yaml.safe_dump(
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
                "drive_upload_enabled": False,
            }
        ),
        encoding="utf-8",
    )

    load_dotenv(dotenv_path=tmp_path / ".env", override=False)
    try:
        cfg = load_config(str(cfg_yaml))
        assert cfg.drive_upload_enabled is True
        assert cfg.drive_credentials_path == "/tmp/from-dotenv-sa.json"
        assert cfg.drive_root_folder_id == "folder-from-dotenv"
    finally:
        for k in ("DRIVE_UPLOAD_ENABLED", "DRIVE_CREDENTIALS_PATH", "DRIVE_ROOT_FOLDER_ID"):
            os.environ.pop(k, None)


def test_target_dates_day_of_week_validation_catches_may_16_2026_saturday() -> None:
    bad = {
        "symbol": "NBIS",
        "today_date": "Tue May 12, 2026",
        "today_session": "regular",
        "earnings_date": "Wed May 13 2026",
        "target_dates": ["Wed May 13", "Thu May 14", "Fri May 16"],
        "next_trading_day": "Thu May 14",
        "followup_open_date": "Wed May 20",
        "providers": ["openai"],
    }
    with pytest.raises(ValueError, match="target_dates entry"):
        RunConfig.model_validate(bad)

    good = {**bad, "target_dates": ["Wed May 13", "Thu May 14", "Fri May 15"]}
    cfg = RunConfig.model_validate(good)
    assert "Fri May 15" in cfg.target_dates


def test_run_profile_defaults_to_dev() -> None:
    cfg = RunConfig.model_validate(_minimal_run_config_dict())
    assert cfg.run_profile == "dev"


def test_run_profile_env_equity_run_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EQUITY_RUN_PROFILE", "production")
    try:
        cfg = RunConfig.model_validate(_minimal_run_config_dict())
        assert cfg.run_profile == "production"
    finally:
        monkeypatch.delenv("EQUITY_RUN_PROFILE", raising=False)


def test_run_profile_env_run_profile_when_equity_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EQUITY_RUN_PROFILE", raising=False)
    monkeypatch.setenv("RUN_PROFILE", "production")
    try:
        cfg = RunConfig.model_validate(_minimal_run_config_dict())
        assert cfg.run_profile == "production"
    finally:
        monkeypatch.delenv("RUN_PROFILE", raising=False)


def test_run_profile_equity_env_wins_over_run_profile_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EQUITY_RUN_PROFILE", "dev")
    monkeypatch.setenv("RUN_PROFILE", "production")
    try:
        cfg = RunConfig.model_validate(_minimal_run_config_dict())
        assert cfg.run_profile == "dev"
    finally:
        monkeypatch.delenv("EQUITY_RUN_PROFILE", raising=False)
        monkeypatch.delenv("RUN_PROFILE", raising=False)


def test_run_profile_invalid_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EQUITY_RUN_PROFILE", "staging")
    try:
        with pytest.raises(ValueError, match="EQUITY_RUN_PROFILE"):
            RunConfig.model_validate(_minimal_run_config_dict())
    finally:
        monkeypatch.delenv("EQUITY_RUN_PROFILE", raising=False)


def test_env_defaults_production() -> None:
    cfg = RunConfig.model_validate(_minimal_run_config_dict())
    assert cfg.env == "production"


def test_env_yaml_test_sets_dev_keeps_db_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DB_ENABLED", raising=False)
    d = _minimal_run_config_dict()
    d["env"] = "test"
    cfg = RunConfig.model_validate(d)
    assert cfg.env == "test"
    assert cfg.run_profile == "dev"
    assert cfg.db_enabled is True


def test_env_yaml_test_keeps_explicit_run_profile() -> None:
    d = _minimal_run_config_dict()
    d["env"] = "test"
    d["run_profile"] = "production"
    cfg = RunConfig.model_validate(d)
    assert cfg.env == "test"
    assert cfg.run_profile == "production"


def test_env_yaml_test_keeps_db_when_yaml_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DB_ENABLED", raising=False)
    d = _minimal_run_config_dict()
    d["env"] = "test"
    d["db_enabled"] = True
    cfg = RunConfig.model_validate(d)
    assert cfg.db_enabled is True


def test_env_yaml_test_respects_db_disabled_in_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DB_ENABLED", raising=False)
    d = _minimal_run_config_dict()
    d["env"] = "test"
    d["db_enabled"] = False
    cfg = RunConfig.model_validate(d)
    assert cfg.db_enabled is False


def test_equity_env_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EQUITY_ENV", "test")
    try:
        cfg = RunConfig.model_validate(_minimal_run_config_dict())
        assert cfg.env == "test"
    finally:
        monkeypatch.delenv("EQUITY_ENV", raising=False)


def test_env_yaml_wins_over_equity_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EQUITY_ENV", "test")
    try:
        d = _minimal_run_config_dict()
        d["env"] = "production"
        cfg = RunConfig.model_validate(d)
        assert cfg.env == "production"
    finally:
        monkeypatch.delenv("EQUITY_ENV", raising=False)


def test_equity_env_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EQUITY_ENV", "staging")
    try:
        with pytest.raises(ValueError, match="EQUITY_ENV"):
            RunConfig.model_validate(_minimal_run_config_dict())
    finally:
        monkeypatch.delenv("EQUITY_ENV", raising=False)


def test_env_yaml_test_overrides_run_profile_from_env_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """``env=test`` resets implicit production from EQUITY_RUN_PROFILE when YAML omits ``run_profile``."""
    monkeypatch.delenv("RUN_PROFILE", raising=False)
    monkeypatch.setenv("EQUITY_RUN_PROFILE", "production")
    try:
        d = _minimal_run_config_dict()
        d["env"] = "test"
        cfg = RunConfig.model_validate(d)
        assert cfg.run_profile == "dev"
    finally:
        monkeypatch.delenv("EQUITY_RUN_PROFILE", raising=False)


def test_t0_blend_preset_defaults() -> None:
    cfg = RunConfig.model_validate(_minimal_run_config_dict())
    assert cfg.t0_blend_preset == "default"


def test_t0_blend_preset_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EQUITY_T0_BLEND_PRESET", "quant_lean")
    try:
        cfg = RunConfig.model_validate(_minimal_run_config_dict())
        assert cfg.t0_blend_preset == "quant_lean"
    finally:
        monkeypatch.delenv("EQUITY_T0_BLEND_PRESET", raising=False)


def test_t0_blend_preset_yaml_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EQUITY_T0_BLEND_PRESET", "quant_lean")
    try:
        d = _minimal_run_config_dict()
        d["t0_blend_preset"] = "qual_dominant"
        cfg = RunConfig.model_validate(d)
        assert cfg.t0_blend_preset == "qual_dominant"
    finally:
        monkeypatch.delenv("EQUITY_T0_BLEND_PRESET", raising=False)


def test_t0_blend_preset_invalid_yaml_raises() -> None:
    d = _minimal_run_config_dict()
    d["t0_blend_preset"] = "bogus"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(d)
