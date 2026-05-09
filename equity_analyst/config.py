from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TextIO

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    symbol: str
    company_name: str | None = None

    today_low: float
    today_high: float
    current_price: float
    today_date: str
    today_session: str

    earnings_date: str
    earnings_timing: str

    target_dates: list[str] = Field(default_factory=list)
    next_trading_day: str
    followup_open_date: str

    historical_quarters: int = 11
    short_interest_lookbacks: list[str] = Field(default_factory=list)

    providers: list[str] = Field(default_factory=lambda: ["anthropic", "openai"])
    synthesizer: str = "anthropic"


def _load_yaml_from_stream(stream: TextIO) -> dict[str, Any]:
    data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping/object")
    return data


def load_config(path_or_dash: str) -> RunConfig:
    if path_or_dash == "-":
        data = _load_yaml_from_stream(stream=sys.stdin)
        return RunConfig.model_validate(data)

    path = Path(path_or_dash)
    data = _load_yaml_from_stream(path.open("r", encoding="utf-8"))
    return RunConfig.model_validate(data)

