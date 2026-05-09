from __future__ import annotations

from pathlib import Path

import yaml

from equity_analyst.cli import _apply_cli_config_overrides, _build_parser, _load_cfg


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
