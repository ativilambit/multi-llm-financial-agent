from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from equity_analyst.cli import _apply_cli_config_overrides, _build_parser, _load_cfg


def test_cli_max_output_tokens_override(tmp_path: Path) -> None:
    yml = tmp_path / "c.yaml"
    yml.write_text(
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
                "max_output_tokens": 24_000,
            }
        ),
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(["run", "--config", str(yml), "--max-output-tokens", "48000"])
    base = _load_cfg(args)
    assert base.max_output_tokens == 24_000
    cfg = _apply_cli_config_overrides(base, args)
    assert cfg.max_output_tokens == 48_000


def test_cli_no_pdf_sets_run_config_false(tmp_path: Path) -> None:
    yml = tmp_path / "c.yaml"
    yml.write_text(
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
            }
        ),
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(["run", "--config", str(yml), "--no-pdf"])
    base = _load_cfg(args)
    assert base.pdf_output_enabled is True
    cfg = _apply_cli_config_overrides(base, args)
    assert cfg.pdf_output_enabled is False


def test_cli_no_prompt_cache_sets_run_config_false(tmp_path: Path) -> None:
    yml = tmp_path / "c.yaml"
    yml.write_text(
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
            }
        ),
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(["run", "--config", str(yml), "--no-prompt-cache"])
    base = _load_cfg(args)
    assert base.prompt_cache_enabled is True
    cfg = _apply_cli_config_overrides(base, args)
    assert cfg.prompt_cache_enabled is False


def test_cli_synthesizer_max_output_tokens_override(tmp_path: Path) -> None:
    yml = tmp_path / "c.yaml"
    yml.write_text(
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
            }
        ),
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(
        ["run", "--config", str(yml), "--synthesizer-max-output-tokens", "50000"]
    )
    assert args.command == "run"
    base = _load_cfg(args)
    assert base.synthesizer_max_output_tokens == 24_000
    cfg = _apply_cli_config_overrides(base, args)
    assert cfg.synthesizer_max_output_tokens == 50_000


def test_cli_summarize_oversized_overrides(tmp_path: Path) -> None:
    yml = tmp_path / "c.yaml"
    yml.write_text(
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
            }
        ),
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(
        [
            "run",
            "--config",
            str(yml),
            "--no-summarize-oversized",
            "--summarize-threshold-tokens",
            "12000",
        ]
    )
    base = _load_cfg(args)
    assert base.summarize_oversized_providers is True
    assert base.summarize_threshold_input_tokens == 8000
    cfg = _apply_cli_config_overrides(base, args)
    assert cfg.summarize_oversized_providers is False
    assert cfg.summarize_threshold_input_tokens == 12_000


def test_cli_drive_upload_and_folder_overrides(tmp_path: Path) -> None:
    yml = tmp_path / "c.yaml"
    yml.write_text(
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
                "drive_root_folder_id": "old",
            }
        ),
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(
        ["run", "--config", str(yml), "--upload-to-drive", "--drive-folder-id", "newroot"]
    )
    base = _load_cfg(args)
    cfg = _apply_cli_config_overrides(base, args)
    assert cfg.drive_upload_enabled is True
    assert cfg.drive_root_folder_id == "newroot"


def test_cli_run_environment_override(tmp_path: Path) -> None:
    yml = tmp_path / "c.yaml"
    yml.write_text(
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
                "run_environment": "production",
            }
        ),
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(["run", "--config", str(yml), "--environment", "test"])
    base = _load_cfg(args)
    cfg = _apply_cli_config_overrides(base, args)
    assert cfg.run_environment == "test"
    assert cfg.env == "test"

    args2 = parser.parse_args(["run", "--config", str(yml), "--environment", "production"])
    cfg2 = _apply_cli_config_overrides(_load_cfg(args2), args2)
    assert cfg2.run_environment == "production"
    assert cfg2.env == "production"


def test_cli_drive_env_alias_matches_environment(tmp_path: Path) -> None:
    yml = tmp_path / "c.yaml"
    yml.write_text(
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
                "run_environment": "production",
            }
        ),
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(["run", "--config", str(yml), "--drive-env", "test"])
    cfg = _apply_cli_config_overrides(_load_cfg(args), args)
    assert cfg.run_environment == "test"
    assert cfg.env == "test"


def test_cli_yaml_may_split_env_and_run_environment_without_cli(tmp_path: Path) -> None:
    yml = tmp_path / "c.yaml"
    yml.write_text(
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
                "env": "production",
                "run_environment": "test",
            }
        ),
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(["run", "--config", str(yml)])
    cfg = _apply_cli_config_overrides(_load_cfg(args), args)
    assert cfg.env == "production"
    assert cfg.run_environment == "test"


def test_cli_env_overrides_both_when_yaml_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EQUITY_ENV", raising=False)
    monkeypatch.delenv("RUN_ENVIRONMENT", raising=False)
    yml = tmp_path / "c.yaml"
    yml.write_text(
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
                "env": "production",
                "run_environment": "test",
            }
        ),
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(["run", "--config", str(yml), "--env", "production"])
    cfg = _apply_cli_config_overrides(_load_cfg(args), args)
    assert cfg.env == "production"
    assert cfg.run_environment == "production"


def test_cli_equity_env_test_sets_dev_keeps_db_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DB_ENABLED", raising=False)
    monkeypatch.delenv("EQUITY_RUN_PROFILE", raising=False)
    monkeypatch.delenv("RUN_PROFILE", raising=False)
    yml = tmp_path / "c.yaml"
    yml.write_text(
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
                "run_profile": "production",
            }
        ),
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(["run", "--config", str(yml), "--env", "test"])
    base = _load_cfg(args)
    assert base.env == "production"
    assert base.run_profile == "production"
    assert base.db_enabled is True
    cfg = _apply_cli_config_overrides(base, args)
    assert cfg.env == "test"
    assert cfg.run_environment == "test"
    assert cfg.run_profile == "dev"
    assert cfg.db_enabled is True


def test_cli_equity_env_test_keeps_profile_when_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DB_ENABLED", raising=False)
    yml = tmp_path / "c.yaml"
    yml.write_text(
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
                "run_profile": "dev",
            }
        ),
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(["run", "--config", str(yml), "--env", "test", "--profile", "production"])
    base = _load_cfg(args)
    cfg = _apply_cli_config_overrides(base, args)
    assert cfg.env == "test"
    assert cfg.run_environment == "test"
    assert cfg.run_profile == "production"


def test_cli_drive_auth_mode_override(tmp_path: Path) -> None:
    yml = tmp_path / "c.yaml"
    yml.write_text(
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
                "drive_auth_mode": "service_account",
            }
        ),
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(["run", "--config", str(yml), "--drive-auth-mode", "oauth_user"])
    base = _load_cfg(args)
    cfg = _apply_cli_config_overrides(base, args)
    assert cfg.drive_auth_mode == "oauth_user"


def test_cli_run_profile_flag_overrides_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("EQUITY_RUN_PROFILE", raising=False)
    monkeypatch.delenv("RUN_PROFILE", raising=False)
    yml = tmp_path / "c.yaml"
    yml.write_text(
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
                "run_profile": "dev",
            }
        ),
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(["run", "--config", str(yml), "--profile", "production"])
    base = _load_cfg(args)
    assert base.run_profile == "dev"
    with caplog.at_level(logging.WARNING, logger="equity_analyst.cli"):
        cfg = _apply_cli_config_overrides(base, args)
    assert cfg.run_profile == "production"
    assert any("CLI --profile is deprecated" in r.message for r in caplog.records)
