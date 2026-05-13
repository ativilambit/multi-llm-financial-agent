"""Parse and validate mandatory ``sigma_summary`` JSON from provider / synthesis text."""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DriftSourceLiteral = Literal[
    "options_skew",
    "PT_consensus",
    "PEAD_avg",
    "recent_momentum",
    "manual_override",
]

# Fenced ```json ... ``` blocks (language tag optional; last matching block wins).
_FENCED_JSON_BLOCKS_RE = re.compile(
    r"```\s*json\s*\r?\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


class SigmaSessionRowModel(BaseModel):
    """One session row inside ``sigma_summary.sessions``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    date: date
    label: str = Field(min_length=1)
    N: int = Field(description="Model-supplied index; verifier recomputes from calendar.")
    one_sigma_half_width_pct: float = Field(gt=0, le=500.0)
    three_sigma_half_width_pct: float = Field(gt=0, le=500.0)
    prob_up_pct: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="P(close > anchor) in percent; server recomputes from bounded drift.",
    )


class SigmaSummaryPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    anchor_price: float = Field(gt=0)
    anchor_type: str = Field(min_length=1)
    daily_drift_pct: float | None = Field(
        default=None,
        description="Signed % drift per trading session; required when emitting probabilities.",
    )
    drift_source: DriftSourceLiteral | None = None
    drift_source_value: str | None = Field(
        default=None,
        min_length=1,
        description="Human-readable underlying number / derivation.",
    )
    drift_citation: str | None = Field(
        default=None,
        min_length=1,
        description="Short provenance string (URL fragment, outcome_tracker, etc.).",
    )
    sessions: list[SigmaSessionRowModel] = Field(min_length=1)

    @model_validator(mode="after")
    def _probabilities_require_drift_bundle(self) -> SigmaSummaryPayloadModel:
        any_prob = any(s.prob_up_pct is not None for s in self.sessions)
        if not any_prob:
            return self
        missing_drift = (
            self.daily_drift_pct is None
            or self.drift_source is None
            or self.drift_source_value is None
            or self.drift_citation is None
        )
        if missing_drift:
            raise ValueError(
                "sigma_summary: when any session sets prob_up_pct, daily_drift_pct, drift_source, "
                "drift_source_value, and drift_citation are required on sigma_summary",
            )
        for s in self.sessions:
            if s.prob_up_pct is None:
                raise ValueError(
                    "sigma_summary: when computing probabilities, every session must include prob_up_pct",
                )
        return self


class SigmaSummaryFileModel(BaseModel):
    """Root object: ``{\"sigma_summary\": {...}}``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    sigma_summary: SigmaSummaryPayloadModel


def parse_sigma_summary_json(text: str) -> SigmaSummaryFileModel | None:
    """Return the **last** fenced ```json`` block that contains ``\"sigma_summary\"``, or ``None``.

    On invalid JSON or schema validation failure, returns ``None`` (caller may fall back to legacy
    markdown parsing for per-provider checks).
    """
    if not text or "sigma_summary" not in text:
        return None
    candidates = list(_FENCED_JSON_BLOCKS_RE.finditer(text))
    for m in reversed(candidates):
        raw = m.group(1).strip()
        if "sigma_summary" not in raw:
            continue
        try:
            data: Any = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        try:
            return SigmaSummaryFileModel.model_validate(data)
        except Exception:
            continue
    return None


def sigma_summary_json_present_but_invalid(text: str) -> bool:
    """True when the **last** ``json`` code fence mentioning ``sigma_summary`` is not schema-valid."""
    if not text or "sigma_summary" not in text:
        return False
    candidates = list(_FENCED_JSON_BLOCKS_RE.finditer(text))
    for m in reversed(candidates):
        raw = m.group(1).strip()
        if "sigma_summary" not in raw:
            continue
        try:
            SigmaSummaryFileModel.model_validate(json.loads(raw))
            return False
        except Exception:
            return True
    return False
