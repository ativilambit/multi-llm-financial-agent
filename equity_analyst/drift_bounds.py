"""Bounded daily drift and deterministic P(up) from σ half-widths (variance-additive context)."""

from __future__ import annotations

import math

DEFAULT_DRIFT_BOUND_PCT_PER_DAY = 0.15
PROB_UP_MISMATCH_TOLERANCE_PP = 2.0

_DRIFT_SOURCES_FROZEN = frozenset(
    {
        "options_skew",
        "PT_consensus",
        "PEAD_avg",
        "recent_momentum",
        "manual_override",
    },
)


def is_valid_drift_source(source: str) -> bool:
    return source.strip() in _DRIFT_SOURCES_FROZEN


def bound_daily_drift(drift_pct: float, source: str) -> tuple[float, str | None]:
    """Clamp drift to default bound unless source explicitly justifies more.

    Returns ``(clamped_drift, warning_or_None)``.
    """
    bound = DEFAULT_DRIFT_BOUND_PCT_PER_DAY
    if source == "PT_consensus":
        bound = 0.30
    if abs(drift_pct) > bound:
        clamped = math.copysign(bound, drift_pct)
        return (
            clamped,
            f"drift {drift_pct:+.3f}%/day clamped to {clamped:+.3f} by source={source} cap",
        )
    return (drift_pct, None)


def computed_prob_up_pct(daily_drift_pct: float, sigma_half_width_pct: float, n: int) -> float:
    """P(S_T+N > anchor) under variance-additive σ with drift μ.

    Uses Φ(μ·N / σ(T+N)). Both μ and σ are in % units (they cancel in the ratio).
    """
    if sigma_half_width_pct <= 0:
        return 50.0
    z = (daily_drift_pct * n) / sigma_half_width_pct
    return float(50.0 * (1.0 + math.erf(z / math.sqrt(2.0))))
