from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal, Self, TextIO, cast

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

KNOWN_PROVIDER_NAMES: frozenset[str] = frozenset({"anthropic", "openai", "gemini", "grok"})

DriveAuthMode = Literal["service_account", "oauth_user"]
RunEnvironment = Literal["production", "test"]

_DEFAULT_OAUTH_CONFIG_DIR = Path.home() / ".config" / "multi-llm-equity-analyst"


def default_oauth_config_dir() -> Path:
    return _DEFAULT_OAUTH_CONFIG_DIR


def resolve_drive_oauth_token_path_from_optional(raw: str | None) -> Path:
    """Resolve OAuth token storage path (YAML/env override or default under ``~/.config/...``)."""
    if raw is not None and str(raw).strip():
        return Path(os.path.expandvars(os.path.expanduser(str(raw).strip()))).resolve()
    return (_DEFAULT_OAUTH_CONFIG_DIR / "oauth_token.json").expanduser().resolve()


def resolve_drive_oauth_token_path(cfg: RunConfig) -> Path:
    """Resolved token path for a loaded :class:`RunConfig`."""
    return resolve_drive_oauth_token_path_from_optional(cfg.drive_oauth_token_path)


def resolve_drive_oauth_client_secrets_path_from_optional(raw: str | None) -> Path:
    """Resolve OAuth Desktop client secrets JSON path."""
    if raw is not None and str(raw).strip():
        return Path(os.path.expandvars(os.path.expanduser(str(raw).strip()))).resolve()
    return (_DEFAULT_OAUTH_CONFIG_DIR / "oauth_client.json").expanduser().resolve()


def resolve_drive_oauth_client_secrets_path(cfg: RunConfig) -> Path:
    """Resolved client secrets path for a loaded :class:`RunConfig`."""
    return resolve_drive_oauth_client_secrets_path_from_optional(cfg.drive_oauth_client_secrets_path)


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
    earnings_timing: str | None = Field(
        default=None,
        description="Optional human-readable earnings call timing (BMO/AMC/etc.). When omitted, the equity "
        "prompt instructs models to verify timing via web_search.",
    )

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

    prediction_extract_enabled: bool = Field(
        default=False,
        description="When True, after a run completes, extract structured prediction horizons from synthesis "
        "into Postgres (extra API cost). CLI ``--extract-predictions`` overrides for a single invocation.",
    )
    prediction_extract_provider: str = Field(
        default="gemini",
        description="Registry key for the synthesis prediction extractor LLM (default: gemini).",
    )
    prediction_extract_model: str = Field(
        default="gemini-3-flash-preview",
        description="Model id for prediction extraction (default fast Flash).",
    )
    prediction_extract_max_output_tokens: int = Field(
        default=2048,
        ge=256,
        le=128_000,
        description="Completion budget for the prediction extractor JSON response.",
    )
    prediction_extract_timeout_s: int = Field(
        default=120,
        ge=1,
        le=900,
        description="Wall-clock timeout (seconds) for the prediction extractor LLM call.",
    )

    max_output_tokens: int = Field(default=16_000, ge=256, le=128_000)
    request_timeout_s: float = Field(default=180.0, gt=0)
    verifier_max_output_tokens: int = Field(
        default=8192,
        ge=256,
        le=32_768,
        description="Completion budget for the iterative verifier JSON response (default 8192).",
    )
    synthesizer_max_output_tokens: int = Field(default=24_000, ge=1024, le=128_000)

    retry_max_attempts: int = Field(default=3, ge=1, le=20)
    retry_base_delay_s: float = Field(default=2.0, gt=0, le=120.0)
    synthesizer_max_input_tokens: int = Field(default=100_000, ge=4_000, le=900_000)

    summarize_oversized_providers: bool = Field(
        default=True,
        description="When True, oversized healthy provider bodies are summarized with Gemini Flash before synthesis.",
    )
    summarize_threshold_input_tokens: int = Field(
        default=8000,
        ge=512,
        description="Per-provider body estimate (len(text)//4) above which pre-synthesis summarization runs; "
        "summarization also runs when the sum of healthy bodies exceeds "
        "max(8000, synthesizer_max_input_tokens - 3000).",
    )
    oversized_summarize_model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini model id for compressing oversized provider outputs (no web search).",
    )
    oversized_summarize_max_output_tokens: int = Field(
        default=8192,
        ge=1024,
        le=128_000,
        description="Max output tokens for the oversized-body summarization call.",
    )
    oversized_summarize_max_input_tokens: int = Field(
        default=100_000,
        ge=4096,
        le=500_000,
        description="Estimated input token budget (len//4) sent to the summarizer; larger bodies are shrunk first.",
    )

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

    pdf_output_enabled: bool = Field(
        default=True,
        description="When True, emit a .pdf next to each primary analysis .md (requires WeasyPrint).",
    )
    delete_checkpoint_after_success: bool = Field(
        default=True,
        description="When True, remove iterative checkpoint.sqlite (+ WAL/SHM/journal) from the run directory "
        "after a successful finalize. Set DELETE_CHECKPOINT_AFTER_SUCCESS=false or use --keep-checkpoint to retain.",
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
    run_environment: RunEnvironment = Field(
        default="production",
        description="Drive upload routing: production uses child folder ``prod``; test uses ``test`` "
        "(created under drive_root_folder_id when uploads run).",
    )
    drive_auth_mode: DriveAuthMode = Field(
        default="service_account",
        description="Google Drive credentials: service account JSON key, or end-user OAuth token file.",
    )
    drive_oauth_client_secrets_path: str | None = Field(
        default=None,
        description="Path to Google OAuth 'Desktop app' client secrets JSON (used by drive_oauth_setup only).",
    )
    drive_oauth_token_path: str | None = Field(
        default=None,
        description="Path to store/load OAuth user refresh token JSON; default under ~/.config/multi-llm-equity-analyst/.",
    )

    db_enabled: bool = Field(
        default=True,
        description="When True, write best-effort structured metadata to Postgres (additive; files remain source of truth).",
    )
    database_url: str | None = Field(
        default=None,
        description="Optional DB connection override; when omitted, uses env DATABASE_URL (loaded via python-dotenv).",
    )

    @field_validator("drive_auth_mode", mode="before")
    @classmethod
    def _normalize_drive_auth_mode(cls, v: Any) -> str:
        if v is None:
            return "service_account"
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("service_account", "oauth_user"):
                return s
        raise ValueError("drive_auth_mode must be 'service_account' or 'oauth_user'")

    @field_validator("verifier_provider")
    @classmethod
    def _known_verifier_provider(cls, v: str) -> str:
        if v not in KNOWN_PROVIDER_NAMES:
            raise ValueError(
                f"Unknown verifier_provider {v!r}. Expected one of: {', '.join(sorted(KNOWN_PROVIDER_NAMES))}"
            )
        return v

    @field_validator("prediction_extract_provider")
    @classmethod
    def _known_prediction_extract_provider(cls, v: str) -> str:
        if v not in KNOWN_PROVIDER_NAMES:
            raise ValueError(
                f"Unknown prediction_extract_provider {v!r}. Expected one of: "
                f"{', '.join(sorted(KNOWN_PROVIDER_NAMES))}"
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
        env_auth = os.environ.get("DRIVE_AUTH_MODE")
        if env_auth is not None and str(env_auth).strip():
            s = str(env_auth).strip().lower()
            if s in ("service_account", "oauth_user"):
                updates["drive_auth_mode"] = s
        if self.drive_oauth_token_path is None:
            tp = os.environ.get("DRIVE_OAUTH_TOKEN_PATH")
            if tp and str(tp).strip():
                updates["drive_oauth_token_path"] = str(tp).strip()
        if self.drive_oauth_client_secrets_path is None:
            cp = os.environ.get("DRIVE_OAUTH_CLIENT_SECRETS_PATH")
            if cp and str(cp).strip():
                updates["drive_oauth_client_secrets_path"] = str(cp).strip()
        return self.model_copy(update=updates) if updates else self

    @model_validator(mode="after")
    def _run_environment_env_override(self) -> Self:
        raw = os.environ.get("RUN_ENVIRONMENT")
        if raw is None or not str(raw).strip():
            return self
        s = str(raw).strip().lower()
        if s not in ("production", "test"):
            raise ValueError(
                "RUN_ENVIRONMENT must be 'production' or 'test' "
                f"(got {raw!r}); unset the variable or fix the value."
            )
        return self.model_copy(update={"run_environment": cast(RunEnvironment, s)})

    @model_validator(mode="after")
    def _pdf_output_env_fallback(self) -> Self:
        env_flag = os.environ.get("PDF_OUTPUT_ENABLED")
        if env_flag is None:
            return self
        enabled = env_flag.strip().lower() in ("1", "true", "yes", "on")
        return self.model_copy(update={"pdf_output_enabled": enabled})

    @model_validator(mode="after")
    def _db_env_fallback(self) -> Self:
        updates: dict[str, Any] = {}
        env_flag = os.environ.get("DB_ENABLED")
        if env_flag is not None and str(env_flag).strip():
            updates["db_enabled"] = env_flag.strip().lower() in ("1", "true", "yes", "on")
        if self.database_url is None:
            raw = os.environ.get("DATABASE_URL")
            if raw is not None and str(raw).strip():
                updates["database_url"] = str(raw).strip()
        return self.model_copy(update=updates) if updates else self

    @model_validator(mode="after")
    def _delete_checkpoint_env_fallback(self) -> Self:
        raw = os.environ.get("DELETE_CHECKPOINT_AFTER_SUCCESS")
        if raw is None or not str(raw).strip():
            return self
        s = str(raw).strip().lower()
        delete_after = s in ("1", "true", "yes", "on")
        return self.model_copy(update={"delete_checkpoint_after_success": delete_after})

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
