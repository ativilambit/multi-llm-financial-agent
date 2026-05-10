from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Self, TextIO

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

KNOWN_PROVIDER_NAMES: frozenset[str] = frozenset({"anthropic", "openai", "gemini", "grok"})


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: str | None = None
    web_search: bool | None = None
    request_timeout_s: float | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, ge=256, le=128_000)

    @field_validator("name")
    @classmethod
    def _known_provider(cls, v: str) -> str:
        if v not in KNOWN_PROVIDER_NAMES:
            raise ValueError(
                f"Unknown provider name {v!r}. Expected one of: {', '.join(sorted(KNOWN_PROVIDER_NAMES))}"
            )
        return v


class SynthesizerConfig(BaseModel):
    """Which backend performs final synthesis (may differ from fan-out providers)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    model: str | None = None
    web_search: bool | None = None
    request_timeout_s: float | None = Field(default=None, gt=0)

    @field_validator("name")
    @classmethod
    def _known_synthesizer(cls, v: str) -> str:
        if v not in KNOWN_PROVIDER_NAMES:
            raise ValueError(
                f"Unknown synthesizer name {v!r}. Expected one of: {', '.join(sorted(KNOWN_PROVIDER_NAMES))}"
            )
        return v


class RunConfig(BaseModel):
    symbol: str
    company_name: str | None = None

    today_low: float | None = Field(
        default=None,
        validation_alias=AliasChoices("today_low", "reference_session_low"),
        description="Optional unverified session low hint; models must verify via web_search. "
        "YAML alias: reference_session_low.",
    )
    today_high: float | None = Field(
        default=None,
        validation_alias=AliasChoices("today_high", "reference_session_high"),
        description="Optional unverified session high hint; models must verify via web_search. "
        "YAML alias: reference_session_high.",
    )
    current_price: float | None = Field(
        default=None,
        validation_alias=AliasChoices("current_price", "reference_last_price"),
        description="Optional unverified last/reference price hint; models must verify via web_search. "
        "YAML alias: reference_last_price.",
    )
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
    synthesizer: SynthesizerConfig = Field(
        default_factory=lambda: SynthesizerConfig(name="gemini"),
    )
    verifier_provider: str = Field(
        default="gemini",
        description="Registry key for the iterative-mode verification LLM (e.g. gemini, anthropic).",
    )
    verifier_model: str | None = Field(
        default=None,
        description="Optional API model id for the verifier; default is each provider's built-in default.",
    )

    max_output_tokens: int = Field(default=16_000, ge=256, le=128_000)
    request_timeout_s: float = Field(default=180.0, gt=0)
    verifier_max_output_tokens: int = Field(default=1536, ge=256, le=32_768)
    synthesizer_max_output_tokens: int = Field(default=24_000, ge=1024, le=128_000)

    retry_max_attempts: int = Field(default=3, ge=1, le=20)
    retry_base_delay_s: float = Field(default=2.0, gt=0, le=120.0)
    synthesizer_max_input_tokens: int = Field(default=20_000, ge=1024, le=500_000)

    prompt_cache_enabled: bool = Field(
        default=True,
        description="When True, Anthropic fan-out uses API prompt caching on system + tools; "
        "Gemini fan-out uses explicit context caching for the static persona (see gemini_cache_ttl_s).",
    )
    gemini_cache_ttl_s: int = Field(
        default=3600,
        ge=60,
        le=86_400,
        description="TTL (seconds) for Gemini explicit context caches when prompt_cache_enabled is True.",
    )
    anthropic_force_tool_use: bool = Field(
        default=True,
        description="When True and web search is enabled for Anthropic, set tool_choice so the model must "
        "use at least one tool (avoids empty refusals when tools are available).",
    )

    drive_upload_enabled: bool = Field(
        default=False,
        description="When True, upload the run output directory to Google Drive after the run completes.",
    )
    drive_credentials_path: str | None = Field(
        default=None,
        description="Path to a Google Cloud service-account JSON key with Drive API access.",
    )
    drive_root_folder_id: str | None = Field(
        default=None,
        description="Google Drive folder ID (under which a per-run subfolder is created).",
    )

    @field_validator("verifier_provider")
    @classmethod
    def _known_verifier_provider(cls, v: str) -> str:
        if v not in KNOWN_PROVIDER_NAMES:
            raise ValueError(
                f"Unknown verifier_provider {v!r}. Expected one of: {', '.join(sorted(KNOWN_PROVIDER_NAMES))}"
            )
        return v

    @field_validator("synthesizer", mode="before")
    @classmethod
    def _coerce_synthesizer(cls, v: Any) -> Any:
        if v is None:
            return {"name": "gemini"}
        if isinstance(v, str):
            return {"name": v}
        if isinstance(v, SynthesizerConfig):
            return v.model_dump()
        if isinstance(v, dict):
            return dict(v)
        raise ValueError(f"Invalid synthesizer entry type: {type(v).__name__}")

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

    @model_validator(mode="after")
    def _providers_non_empty(self) -> Self:
        if not self.providers:
            raise ValueError("providers must contain at least one entry")
        return self

    @model_validator(mode="after")
    def _drive_env_fallback(self) -> Self:
        updates: dict[str, Any] = {}
        env_flag = os.environ.get("DRIVE_UPLOAD_ENABLED")
        if env_flag is not None:
            updates["drive_upload_enabled"] = env_flag.strip().lower() in ("1", "true", "yes", "on")
        if self.drive_credentials_path is None:
            p = os.environ.get("DRIVE_CREDENTIALS_PATH")
            if p and str(p).strip():
                updates["drive_credentials_path"] = str(p).strip()
        if self.drive_root_folder_id is None:
            f = os.environ.get("DRIVE_ROOT_FOLDER_ID")
            if f and str(f).strip():
                updates["drive_root_folder_id"] = str(f).strip()
        return self.model_copy(update=updates) if updates else self

    def provider_names(self) -> list[str]:
        return [p.name for p in self.providers]

    def synthesizer_timeout_s(self) -> float:
        if self.synthesizer.request_timeout_s is not None:
            return float(self.synthesizer.request_timeout_s)
        return float(self.request_timeout_s)


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
