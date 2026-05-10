from __future__ import annotations

from pathlib import Path

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
            }
        ),
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(["run", "--config", str(yml), "--max-output-tokens", "32000"])
    base = _load_cfg(args)
    assert base.max_output_tokens == 16_000
    cfg = _apply_cli_config_overrides(base, args)
    assert cfg.max_output_tokens == 32_000


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
