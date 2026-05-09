from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TextIO

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    web_search: bool | None = None
    request_timeout_s: float | None = Field(default=None, gt=0)


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

    providers: list[ProviderConfig] = Field(
        default_factory=lambda: [
            ProviderConfig(name="anthropic"),
            ProviderConfig(name="openai"),
        ]
    )
    synthesizer: str = "anthropic"

    max_output_tokens: int = Field(default=4096, ge=256, le=128_000)
    request_timeout_s: float = Field(default=180.0, gt=0)
    verifier_max_output_tokens: int = Field(default=1536, ge=256, le=32_768)

    @field_validator("providers", mode="before")
    @classmethod
    def _coerce_providers(cls, v: Any) -> Any:
        if v is None:
            return ["anthropic", "openai"]
        if not isinstance(v, list):
            raise ValueError("providers must be a list")
        out: list[Any] = []
        for item in v:
            if isinstance(item, ProviderConfig):
                out.append(item.model_dump())
            elif isinstance(item, str):
                out.append({"name": item})
            elif isinstance(item, dict):
                out.append(dict(item))
            else:
                raise ValueError(f"Invalid provider entry type: {type(item).__name__}")
        return out

    def provider_names(self) -> list[str]:
        return [p.name for p in self.providers]


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
