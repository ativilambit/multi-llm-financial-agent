from __future__ import annotations

import calendar
import logging
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Literal, Self, TextIO, cast

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from equity_analyst.synthesizer_blend import T0BlendPreset, normalize_t0_blend_preset

KNOWN_PROVIDER_NAMES: frozenset[str] = frozenset({"anthropic", "openai", "gemini", "grok"})

logger = logging.getLogger(__name__)

_TARGET_DATE_DOW_PREFIX_RE = re.compile(
    r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
    r"Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b[,\s]+(.+)$",
    re.IGNORECASE,
)


def _weekday_index_from_label(label: str) -> int | None:
    low = label.strip().lower()
    if len(low) < 3:
        return None
    key = low[:3]
    mapping = {
        "mon": 0,
        "tue": 1,
        "wed": 2,
        "thu": 3,
        "fri": 4,
        "sat": 5,
        "sun": 6,
    }
    return mapping.get(key)

_FACTS_PACKET_MAX_OUT_MIN = 256
_FACTS_PACKET_MAX_OUT_MAX = 128_000


def _env_flag_truthy(name: str) -> bool:
    """Return True when ``os.environ[name]`` is a common affirmative string."""
    v = os.environ.get(name, "")
    return v.strip().lower() in ("1", "true", "yes", "on")


def _parse_facts_packet_max_output_tokens_env(raw: str) -> int | None:
    """Parse ``FACTS_PACKET_MAX_OUTPUT_TOKENS``; return ``None`` if missing or invalid."""
    s = str(raw).strip()
    if not s:
        return None
    try:
        n = int(s, 10)
    except ValueError:
        return None
    if _FACTS_PACKET_MAX_OUT_MIN <= n <= _FACTS_PACKET_MAX_OUT_MAX:
        return n
    return None


_VERIFIER_MAX_OUT_MIN = 256
_VERIFIER_MAX_OUT_MAX = 32_768


def _sigma_variance_check_quorum_for_error_env_fallback(raw: str) -> int | None:
    """Parse ``SIGMA_VARIANCE_CHECK_QUORUM_FOR_ERROR``; return ``None`` if missing or invalid."""
    s = str(raw).strip()
    if not s:
        return None
    try:
        n = int(s, 10)
    except ValueError:
        return None
    if 1 <= n <= 10:
        return n
    return None


def _parse_verifier_max_output_tokens_env(raw: str) -> int | None:
    """Parse ``VERIFIER_MAX_OUTPUT_TOKENS``; return ``None`` if missing or invalid."""
    s = str(raw).strip()
    if not s:
        return None
    try:
        n = int(s, 10)
    except ValueError:
        return None
    if _VERIFIER_MAX_OUT_MIN <= n <= _VERIFIER_MAX_OUT_MAX:
        return n
    return None


def _parse_max_weekly_lookahead_days_env(raw: str | None) -> int | None:
    """Parse ``MAX_WEEKLY_LOOKAHEAD_DAYS`` / ``EQUITY_MAX_WEEKLY_LOOKAHEAD_DAYS``."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        n = int(s, 10)
    except ValueError:
        return None
    if 1 <= n <= 120:
        return n
    return None


def _default_max_weekly_lookahead_days() -> int:
    v = _parse_max_weekly_lookahead_days_env(os.environ.get("MAX_WEEKLY_LOOKAHEAD_DAYS"))
    if v is not None:
        return v
    v2 = _parse_max_weekly_lookahead_days_env(os.environ.get("EQUITY_MAX_WEEKLY_LOOKAHEAD_DAYS"))
    return v2 if v2 is not None else 14


DriveAuthMode = Literal["service_account", "oauth_user"]
RunEnvironment = Literal["production", "test"]
RunProfile = Literal["production", "dev"]

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

    same_day_intraday_min: float | None = Field(
        default=None,
        description="Optional same-trading-day session low (USD) for earnings_date; injected into the equity "
        "template as same_day_intraday_min when paired with same_day_intraday_max.",
    )
    same_day_intraday_max: float | None = Field(
        default=None,
        description="Optional same-trading-day session high (USD) for earnings_date; injected when paired "
        "with same_day_intraday_min.",
    )
    same_day_intraday_auto_fetch: bool = Field(
        default_factory=lambda: _env_flag_truthy("SAME_DAY_INTRADAY_AUTO_FETCH"),
        description="When True and same_day_intraday_min/max are unset, render_prompt attempts Yahoo Finance "
        "(via fetch_earnings_day_intraday_high_low_yfinance) to populate same-day bounds. Enable with env "
        "SAME_DAY_INTRADAY_AUTO_FETCH=1 for intraday/post-close runs.",
    )

    options_chain_auto_fetch: bool = Field(
        default=True,
        description="When True and options_chain_snapshot is unset, render_prompt fetches a Yahoo option chain "
        "via yfinance (may rate-limit). Opt out globally with OPTIONS_CHAIN_AUTO_FETCH=0.",
    )
    options_chain_snapshot: dict[str, Any] | None = Field(
        default=None,
        description="Optional manual override: mapping matching OptionsChainSnapshot.to_prompt_dict(); "
        "when set, fetch is skipped.",
    )
    max_weekly_lookahead_days: int = Field(
        default_factory=_default_max_weekly_lookahead_days,
        ge=1,
        le=120,
        description="Prefer a non-monthly (weekly-like) listed expiry within this many calendar days after "
        "earnings for the front straddle; otherwise fall back to the nearest standard monthly (3rd Friday). "
        "CLI ``--max-weekly-lookahead-days`` / env ``MAX_WEEKLY_LOOKAHEAD_DAYS`` or ``EQUITY_MAX_WEEKLY_LOOKAHEAD_DAYS``.",
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

    facts_packet_enabled: bool = Field(
        default=True,
        description="Iterative: after round-1 synthesis, extract a compact facts_packet.md and prepend it "
        "on later fan-out rounds to reduce duplicate web fetches.",
    )
    conditional_fanout_enabled: bool = Field(
        default=True,
        description="Iterative: after round 1, skip fan-out providers unless the verifier requests re-fan-out "
        "or (when fan_out_on_continue is True) the router emitted follow-up questions for this round.",
    )
    fan_out_on_continue: bool = Field(
        default=True,
        description="Iterative: when True, iteration 2+ runs full provider fan-out if the route step appended "
        "follow-up questions (from contradictions / unverifiable), even when conditional_fanout_enabled is True "
        "and the verifier left refan_out_* unset. Set False to keep legacy cost behavior (verifier refan only).",
    )
    unverifiable_only_skip_fan_out: bool = Field(
        default=True,
        description="Iterative: when True and the verifier reports contradictions=0 with only unverifiable items, "
        "route to synthesize+verify again without re-calling fan-out providers (cheaper citation cleanup).",
    )
    unverifiable_count_threshold_for_fanout: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Iterative: when unverifiable count is at least this value and overall_confidence is below "
        "unverifiable_fanout_confidence_below, re-run provider fan-out even if there are no contradictions.",
    )
    unverifiable_fanout_confidence_below: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Iterative: paired with unverifiable_count_threshold_for_fanout to trigger a provider re-fan-out "
        "when many items are unverifiable and confidence is low.",
    )
    force_fan_out_on_continue: bool = Field(
        default=False,
        description="Iterative: when True, any router 'continue' goes to provider fan-out (overrides "
        "unverifiable_only_skip_fan_out verify-only routing).",
    )
    refinement_mode_prompt_enabled: bool = Field(
        default=True,
        description="Iterative: when iteration 2+ actually invokes fan-out providers, prepend REFINEMENT MODE "
        "instructions plus prior-round synthesis so models quote FACTS and avoid re-deriving frozen primitives.",
    )
    facts_packet_extractor_provider: str = Field(
        default="gemini",
        description="Registry key for the facts-packet extractor LLM (default fast/cheap).",
    )
    facts_packet_extractor_model: str = Field(
        default="gemini-3-flash-preview",
        description="Model id for facts-packet extraction.",
    )
    facts_packet_max_output_tokens: int = Field(
        default=4096,
        ge=256,
        le=128_000,
        description="Completion budget for facts-packet extraction markdown.",
    )

    max_output_tokens: int = Field(default=16_000, ge=256, le=128_000)
    request_timeout_s: float = Field(default=180.0, gt=0)
    verifier_max_output_tokens: int = Field(
        default=16_384,
        ge=256,
        le=32_768,
        description="Completion budget for the iterative verifier JSON response (default 16384).",
    )
    synthesizer_max_output_tokens: int = Field(default=24_000, ge=1024, le=128_000)

    retry_max_attempts: int = Field(default=3, ge=1, le=20)
    retry_max_attempts_fan_out: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max API retries per provider in iterative fan_out only; other nodes use retry_max_attempts.",
    )
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
    oversized_summarize_provider: str = Field(
        default="gemini",
        description="Registry key for the pre-synthesis oversized-body summarizer (default Gemini Flash API).",
    )
    oversized_summarize_model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini Flash model id for compressing oversized provider outputs (no web search).",
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
    oversized_summarize_min_retention: float = Field(
        default=0.40,
        ge=0.05,
        le=0.95,
        description="If estimated retention (summary vs input len//4) is below this after Gemini summarization, "
        "run one floor-strict retry; optionally try oversized_summarize_fallback_provider.",
    )
    oversized_summarize_fallback_provider: str | None = Field(
        default=None,
        description="Optional provider registry name (e.g. openai) used once if Gemini + retry still miss "
        "oversized_summarize_min_retention. Must match a configured providers[].name.",
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
    final_report_full_synthesis: bool = Field(
        default=True,
        description="Iterative finalize: when True, the iteration changelog in synthesis.md uses each round's "
        "full synthesis text instead of an abridged preview with a pointer to iterations/*.md. "
        "Set FINAL_REPORT_FULL_SYNTHESIS=0 to restore the legacy abridged changelog.",
    )
    delete_checkpoint_after_success: bool = Field(
        default=True,
        description="When True, remove iterative checkpoint.sqlite (+ WAL/SHM/journal) from the run directory "
        "after a successful finalize. Set DELETE_CHECKPOINT_AFTER_SUCCESS=false or use --keep-checkpoint to retain.",
    )
    per_provider_sigma_variance_check: bool = Field(
        default=True,
        description="When True (default), iterative fan-out runs deterministic per-provider sigma-band "
        "variance checks (sigma_summary JSON with legacy markdown fallback). Set False to skip checks "
        "and log once per run.",
    )
    sigma_variance_check_quorum_for_error: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Iterative: per-round quorum — at least this many providers must fail the sigma variance "
        "identity (passed=False while applicable) or omit mandatory literals (missing_literals) before "
        "router emits fan-out follow-ups for that signal. Isolated single-provider misses are severity "
        "warning only (logged + synthesizer table). Set to 1 to treat any single failure as routing error.",
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

    run_profile: RunProfile = Field(
        default="dev",
        description="Only ``production`` runs are persisted to Postgres (runs, provider_responses, outcomes, "
        "predictions). ``dev`` keeps file artifacts only. Override with CLI ``--profile`` or env "
        "``EQUITY_RUN_PROFILE`` / ``RUN_PROFILE``.",
    )
    env: RunEnvironment = Field(
        default="production",
        description="Deployment tier for a run: ``test`` selects dev-safe defaults (``run_profile=dev`` when not "
        "overridden by CLI ``--profile``, and Postgres off unless YAML ``db_enabled: true`` or ``DB_ENABLED=1``). "
        "Separate from ``run_environment`` (Drive upload folder routing). YAML ``env``, env ``EQUITY_ENV``, or "
        "CLI ``--env`` (CLI wins).",
    )
    t0_blend_preset: T0BlendPreset = Field(
        default="default",
        description="T-0-only horizon blend (qual : quant digits). T-3..T-1 stays 55:45; T+1..T+5 stays 49:51. "
        "Override with CLI ``--t0-blend`` or env ``EQUITY_T0_BLEND_PRESET`` when not set in YAML.",
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

    @field_validator("t0_blend_preset", mode="before")
    @classmethod
    def _normalize_t0_blend_preset(cls, v: Any) -> T0BlendPreset:
        return normalize_t0_blend_preset(v)

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

    @field_validator("facts_packet_extractor_provider")
    @classmethod
    def _known_facts_packet_extractor_provider(cls, v: str) -> str:
        if v not in KNOWN_PROVIDER_NAMES:
            raise ValueError(
                f"Unknown facts_packet_extractor_provider {v!r}. Expected one of: "
                f"{', '.join(sorted(KNOWN_PROVIDER_NAMES))}"
            )
        return v

    @field_validator("oversized_summarize_provider")
    @classmethod
    def _known_oversized_summarize_provider(cls, v: str) -> str:
        if v not in KNOWN_PROVIDER_NAMES:
            raise ValueError(
                f"Unknown oversized_summarize_provider {v!r}. Expected one of: "
                f"{', '.join(sorted(KNOWN_PROVIDER_NAMES))}"
            )
        return v

    @field_validator("oversized_summarize_fallback_provider", mode="before")
    @classmethod
    def _normalize_oversized_summarize_fallback_provider(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        if not isinstance(v, str):
            raise ValueError("oversized_summarize_fallback_provider must be a string or null")
        return v.strip()

    @field_validator("oversized_summarize_fallback_provider")
    @classmethod
    def _known_oversized_summarize_fallback_provider(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if v not in KNOWN_PROVIDER_NAMES:
            raise ValueError(
                f"Unknown oversized_summarize_fallback_provider {v!r}. Expected one of: "
                f"{', '.join(sorted(KNOWN_PROVIDER_NAMES))}"
            )
        return v

    @field_validator("run_profile", mode="before")
    @classmethod
    def _normalize_run_profile(cls, v: Any) -> str:
        if v is None:
            return "dev"
        if not isinstance(v, str):
            raise ValueError("run_profile must be a string")
        s = v.strip().lower()
        if s not in ("production", "dev"):
            raise ValueError("run_profile must be 'production' or 'dev'")
        return s

    @field_validator("env", mode="before")
    @classmethod
    def _normalize_env_tier(cls, v: Any) -> str:
        if v is None:
            return "production"
        if not isinstance(v, str):
            raise ValueError("env must be a string")
        s = v.strip().lower()
        if s not in ("production", "test"):
            raise ValueError("env must be 'production' or 'test'")
        return s

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
    def _oversized_summarize_fallback_matches_configured_provider(self) -> Self:
        fb = self.oversized_summarize_fallback_provider
        if fb is None:
            return self
        names = self.provider_names()
        if fb not in names:
            raise ValueError(
                f"oversized_summarize_fallback_provider {fb!r} must match a configured providers[].name; "
                f"got providers={names!r}"
            )
        return self

    @model_validator(mode="after")
    def _providers_non_empty(self) -> Self:
        if not self.providers:
            raise ValueError("providers must contain at least one entry")
        return self

    @model_validator(mode="after")
    def _same_day_intraday_pair(self) -> Self:
        lo, hi = self.same_day_intraday_min, self.same_day_intraday_max
        if (lo is None) ^ (hi is None):
            raise ValueError(
                "same_day_intraday_min and same_day_intraday_max must both be set or both omitted"
            )
        return self

    @model_validator(mode="after")
    def _target_dates_weekday_matches_calendar(self) -> Self:
        """Reject labels like ``Fri May 16`` when that calendar day is not a Friday for the inferred year."""
        if not self.target_dates:
            return self
        from equity_analyst.outcome_tracker import _parse_earnings_date_fuzzy

        year_candidates: list[int] = []
        for hint in (
            self.earnings_date,
            self.today_date,
            self.next_trading_day,
            self.followup_open_date,
        ):
            parsed = _parse_earnings_date_fuzzy(str(hint))
            if parsed is not None and parsed.year >= 1990:
                year_candidates.append(parsed.year)
        year = year_candidates[0] if year_candidates else date.today().year

        for raw in self.target_dates:
            s = str(raw).strip()
            if not s:
                continue
            m = _TARGET_DATE_DOW_PREFIX_RE.match(s)
            if not m:
                continue
            dow_label, rest = m.group(1), m.group(2).strip().strip(",")
            claimed = _weekday_index_from_label(dow_label)
            if claimed is None:
                continue
            blob = rest if re.search(r"\b(19|20)\d{2}\b", rest) else f"{rest} {year}"
            dt = _parse_earnings_date_fuzzy(blob)
            if dt is None:
                continue
            actual = int(dt.weekday())
            if actual != claimed:
                cal_name = calendar.day_name[actual]
                raise ValueError(
                    f"target_dates entry {s!r}: leading weekday does not match calendar date "
                    f"{dt.date().isoformat()} ({cal_name}) for parsed `{blob}`; fix the weekday label or the day."
                )
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
    def _equity_env_env_fallback(self) -> Self:
        """Env ``EQUITY_ENV`` when ``env`` was not set explicitly in YAML (YAML > env > default)."""
        if "env" in self.model_fields_set:
            return self
        raw = os.environ.get("EQUITY_ENV")
        if raw is None or not str(raw).strip():
            return self
        s = str(raw).strip().lower()
        if s not in ("production", "test"):
            raise ValueError(
                "EQUITY_ENV must be 'production' or 'test' "
                f"(got {raw!r}); unset the variable or fix the value."
            )
        return self.model_copy(update={"env": cast(RunEnvironment, s)})

    @model_validator(mode="after")
    def _run_profile_env_fallback(self) -> Self:
        """Env override for ``run_profile`` (same idea as ``DB_ENABLED`` on ``_db_env_fallback``)."""
        if self.env == "test":
            # Test tier: profile comes from YAML / field defaults only; never promote from env vars.
            return self
        raw = os.environ.get("EQUITY_RUN_PROFILE")
        if raw is None or not str(raw).strip():
            raw = os.environ.get("RUN_PROFILE")
        if raw is None or not str(raw).strip():
            return self
        s = str(raw).strip().lower()
        if s not in ("production", "dev"):
            raise ValueError(
                "EQUITY_RUN_PROFILE / RUN_PROFILE must be 'production' or 'dev' "
                f"(got {raw!r}); unset the variable or fix the value."
            )
        return self.model_copy(update={"run_profile": cast(RunProfile, s)})

    @model_validator(mode="after")
    def _env_tier_test_safe_defaults(self) -> Self:
        """When ``env`` is ``test``, prefer dev profile and no Postgres unless explicitly opted in."""
        if self.env != "test":
            return self
        updates: dict[str, Any] = {}
        if "run_profile" not in self.model_fields_set:
            updates["run_profile"] = "dev"
        explicit_db = ("db_enabled" in self.model_fields_set and self.db_enabled is True) or (
            os.environ.get("DB_ENABLED") is not None
            and str(os.environ.get("DB_ENABLED", "")).strip() != ""
            and _env_flag_truthy("DB_ENABLED")
        )
        if not explicit_db:
            updates["db_enabled"] = False
        return self.model_copy(update=updates) if updates else self

    @model_validator(mode="after")
    def _t0_blend_preset_env_fallback(self) -> Self:
        """Env override when ``t0_blend_preset`` was not set explicitly in YAML (YAML > env > default)."""
        if "t0_blend_preset" in self.model_fields_set:
            return self
        raw = os.environ.get("EQUITY_T0_BLEND_PRESET")
        if raw is None or not str(raw).strip():
            return self
        return self.model_copy(update={"t0_blend_preset": normalize_t0_blend_preset(raw)})

    @model_validator(mode="after")
    def _delete_checkpoint_env_fallback(self) -> Self:
        raw = os.environ.get("DELETE_CHECKPOINT_AFTER_SUCCESS")
        if raw is None or not str(raw).strip():
            return self
        s = str(raw).strip().lower()
        delete_after = s in ("1", "true", "yes", "on")
        return self.model_copy(update={"delete_checkpoint_after_success": delete_after})

    @model_validator(mode="after")
    def _final_report_full_synthesis_env_fallback(self) -> Self:
        """Env override when ``final_report_full_synthesis`` was not set explicitly in YAML (YAML > env > default)."""
        raw = os.environ.get("FINAL_REPORT_FULL_SYNTHESIS")
        if (
            raw is None
            or not str(raw).strip()
            or "final_report_full_synthesis" in self.model_fields_set
        ):
            return self
        full = str(raw).strip().lower() in ("1", "true", "yes", "on")
        return self.model_copy(update={"final_report_full_synthesis": full})

    @model_validator(mode="after")
    def _facts_packet_and_conditional_fanout_env_fallback(self) -> Self:
        """Env overrides only when the field was not set explicitly in YAML (YAML > env > default)."""
        updates: dict[str, Any] = {}
        fp = os.environ.get("FACTS_PACKET_ENABLED")
        if (
            fp is not None
            and str(fp).strip()
            and "facts_packet_enabled" not in self.model_fields_set
        ):
            updates["facts_packet_enabled"] = str(fp).strip().lower() in ("1", "true", "yes", "on")
        cf = os.environ.get("CONDITIONAL_FANOUT_ENABLED")
        if (
            cf is not None
            and str(cf).strip()
            and "conditional_fanout_enabled" not in self.model_fields_set
        ):
            updates["conditional_fanout_enabled"] = str(cf).strip().lower() in ("1", "true", "yes", "on")
        foc = os.environ.get("FAN_OUT_ON_CONTINUE")
        if (
            foc is not None
            and str(foc).strip()
            and "fan_out_on_continue" not in self.model_fields_set
        ):
            updates["fan_out_on_continue"] = str(foc).strip().lower() in ("1", "true", "yes", "on")
        rm = os.environ.get("REFINEMENT_MODE_PROMPT_ENABLED")
        if (
            rm is not None
            and str(rm).strip()
            and "refinement_mode_prompt_enabled" not in self.model_fields_set
        ):
            updates["refinement_mode_prompt_enabled"] = str(rm).strip().lower() in ("1", "true", "yes", "on")
        ocf = os.environ.get("OPTIONS_CHAIN_AUTO_FETCH")
        if (
            ocf is not None
            and str(ocf).strip()
            and "options_chain_auto_fetch" not in self.model_fields_set
        ):
            raw = str(ocf).strip().lower()
            if raw in ("1", "true", "yes", "on"):
                updates["options_chain_auto_fetch"] = True
            elif raw in ("0", "false", "no", "off"):
                updates["options_chain_auto_fetch"] = False
            else:
                logger.warning(
                    "Invalid OPTIONS_CHAIN_AUTO_FETCH=%r (expected 1/true/yes/on or 0/false/no/off); "
                    "keeping default/config value.",
                    ocf,
                )
        raw_m = os.environ.get("FACTS_PACKET_MAX_OUTPUT_TOKENS")
        if (
            raw_m is not None
            and str(raw_m).strip()
            and "facts_packet_max_output_tokens" not in self.model_fields_set
        ):
            n = _parse_facts_packet_max_output_tokens_env(str(raw_m))
            if n is not None:
                updates["facts_packet_max_output_tokens"] = n
            else:
                logger.warning(
                    "Invalid FACTS_PACKET_MAX_OUTPUT_TOKENS=%r (expected integer in %s-%s); "
                    "using default %s.",
                    raw_m,
                    _FACTS_PACKET_MAX_OUT_MIN,
                    _FACTS_PACKET_MAX_OUT_MAX,
                    self.facts_packet_max_output_tokens,
                )
        return self.model_copy(update=updates) if updates else self

    @model_validator(mode="after")
    def _sigma_variance_check_quorum_for_error_env_fallback_validator(self) -> Self:
        """Env override when ``sigma_variance_check_quorum_for_error`` was not set explicitly in YAML."""
        raw = os.environ.get("SIGMA_VARIANCE_CHECK_QUORUM_FOR_ERROR")
        if (
            raw is None
            or not str(raw).strip()
            or "sigma_variance_check_quorum_for_error" in self.model_fields_set
        ):
            return self
        n = _sigma_variance_check_quorum_for_error_env_fallback(str(raw))
        if n is not None:
            return self.model_copy(update={"sigma_variance_check_quorum_for_error": n})
        logger.warning(
            "Invalid SIGMA_VARIANCE_CHECK_QUORUM_FOR_ERROR=%r (expected integer 1-10); using default %s.",
            raw,
            self.sigma_variance_check_quorum_for_error,
        )
        return self

    @model_validator(mode="after")
    def _retry_max_attempts_fan_out_env_fallback(self) -> Self:
        """Env override when ``retry_max_attempts_fan_out`` was not set explicitly in YAML."""
        raw = os.environ.get("RETRY_MAX_ATTEMPTS_FAN_OUT")
        if (
            raw is None
            or not str(raw).strip()
            or "retry_max_attempts_fan_out" in self.model_fields_set
        ):
            return self
        try:
            n = int(str(raw).strip())
        except ValueError:
            logger.warning(
                "Invalid RETRY_MAX_ATTEMPTS_FAN_OUT=%r (expected integer 1-20); using default %s.",
                raw,
                self.retry_max_attempts_fan_out,
            )
            return self
        if n < 1 or n > 20:
            logger.warning(
                "Invalid RETRY_MAX_ATTEMPTS_FAN_OUT=%s (expected 1-20); using default %s.",
                n,
                self.retry_max_attempts_fan_out,
            )
            return self
        return self.model_copy(update={"retry_max_attempts_fan_out": n})

    @model_validator(mode="after")
    def _verifier_max_output_tokens_env_fallback(self) -> Self:
        """Env override when ``verifier_max_output_tokens`` was not set explicitly in YAML."""
        raw_m = os.environ.get("VERIFIER_MAX_OUTPUT_TOKENS")
        if (
            raw_m is None
            or not str(raw_m).strip()
            or "verifier_max_output_tokens" in self.model_fields_set
        ):
            return self
        n = _parse_verifier_max_output_tokens_env(str(raw_m))
        if n is not None:
            return self.model_copy(update={"verifier_max_output_tokens": n})
        logger.warning(
            "Invalid VERIFIER_MAX_OUTPUT_TOKENS=%r (expected integer in %s-%s); using default %s.",
            raw_m,
            _VERIFIER_MAX_OUT_MIN,
            _VERIFIER_MAX_OUT_MAX,
            self.verifier_max_output_tokens,
        )
        return self

    @model_validator(mode="after")
    def _oversized_summarize_env_fallback(self) -> Self:
        """Env overrides when the field was not set explicitly in YAML (YAML > env > default)."""
        updates: dict[str, Any] = {}
        raw_p = os.environ.get("OVERSIZED_SUMMARIZE_PROVIDER")
        if (
            raw_p is not None
            and str(raw_p).strip()
            and "oversized_summarize_provider" not in self.model_fields_set
        ):
            updates["oversized_summarize_provider"] = str(raw_p).strip()
        raw_m = os.environ.get("OVERSIZED_SUMMARIZE_MODEL")
        if (
            raw_m is not None
            and str(raw_m).strip()
            and "oversized_summarize_model" not in self.model_fields_set
        ):
            updates["oversized_summarize_model"] = str(raw_m).strip()
        raw_mr = os.environ.get("OVERSIZED_SUMMARIZE_MIN_RETENTION")
        if (
            raw_mr is not None
            and str(raw_mr).strip()
            and "oversized_summarize_min_retention" not in self.model_fields_set
        ):
            try:
                f = float(str(raw_mr).strip())
            except ValueError:
                f = None
            if f is not None and 0.05 <= f <= 0.95:
                updates["oversized_summarize_min_retention"] = f
            elif f is not None:
                logger.warning(
                    "Invalid OVERSIZED_SUMMARIZE_MIN_RETENTION=%r (expected float in 0.05-0.95); using default %s.",
                    raw_mr,
                    self.oversized_summarize_min_retention,
                )
        raw_fb = os.environ.get("OVERSIZED_SUMMARIZE_FALLBACK_PROVIDER")
        if (
            raw_fb is not None
            and str(raw_fb).strip()
            and "oversized_summarize_fallback_provider" not in self.model_fields_set
        ):
            name = str(raw_fb).strip()
            if name in KNOWN_PROVIDER_NAMES and name in self.provider_names():
                updates["oversized_summarize_fallback_provider"] = name
            elif name in KNOWN_PROVIDER_NAMES:
                logger.warning(
                    "OVERSIZED_SUMMARIZE_FALLBACK_PROVIDER=%r is not a configured providers[].name; ignoring.",
                    name,
                )
            else:
                logger.warning(
                    "OVERSIZED_SUMMARIZE_FALLBACK_PROVIDER=%r is not a known provider name; ignoring.",
                    name,
                )
        return self.model_copy(update=updates) if updates else self

    def provider_names(self) -> list[str]:
        return [p.name for p in self.providers]

    def synthesizer_timeout_s(self) -> float:
        if self.synthesizer.request_timeout_s is not None:
            return float(self.synthesizer.request_timeout_s)
        return float(self.request_timeout_s)


def db_enabled_explicitly_opted_in(*, base_cfg: RunConfig) -> bool:
    """True when Postgres should stay enabled for ``env=test`` (YAML or ``DB_ENABLED``)."""
    if "db_enabled" in base_cfg.model_fields_set and base_cfg.db_enabled is True:
        return True
    raw = os.environ.get("DB_ENABLED")
    return bool(raw is not None and str(raw).strip() != "" and _env_flag_truthy("DB_ENABLED"))


def apply_cli_env_tier_overrides(
    merged: RunConfig,
    *,
    base_cfg: RunConfig,
    run_profile_cli: str | None,
    no_db: bool,
) -> RunConfig:
    """Re-apply ``env=test`` defaults after CLI patches (``model_copy`` does not re-run validators)."""
    if merged.env != "test":
        return merged
    updates: dict[str, Any] = {}
    if run_profile_cli is None:
        updates["run_profile"] = "dev"
    if not no_db and not db_enabled_explicitly_opted_in(base_cfg=base_cfg):
        updates["db_enabled"] = False
    return merged.model_copy(update=updates) if updates else merged


def run_profile_from_persisted_run_json(data: dict[str, Any]) -> RunProfile:
    """Resolve Postgres persistence profile from on-disk ``run.json``.

    Precedence: top-level ``run_profile``, then ``config["run_profile"]``.
    Legacy snapshots with no explicit profile are treated as ``production`` so
    historical batch runs keep Postgres outcome/prediction compatibility.
    """
    top = data.get("run_profile")
    if isinstance(top, str) and top.strip().lower() in ("production", "dev"):
        return cast(RunProfile, top.strip().lower())
    cfg = data.get("config")
    if isinstance(cfg, dict):
        v = cfg.get("run_profile")
        if isinstance(v, str) and v.strip().lower() in ("production", "dev"):
            return cast(RunProfile, v.strip().lower())
    return "production"


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
