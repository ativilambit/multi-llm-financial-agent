from __future__ import annotations

import math

import pytest

from equity_analyst.drift_bounds import (
    DEFAULT_DRIFT_BOUND_PCT_PER_DAY,
    bound_daily_drift,
    computed_prob_up_pct,
)


def test_bound_daily_drift_clamps_at_default_cap() -> None:
    d, w = bound_daily_drift(0.5, "PEAD_avg")
    assert d == pytest.approx(DEFAULT_DRIFT_BOUND_PCT_PER_DAY)
    assert w is not None and "clamped" in w


def test_bound_daily_drift_pt_consensus_higher_cap() -> None:
    d, w = bound_daily_drift(0.25, "PT_consensus")
    assert d == pytest.approx(0.25)
    assert w is None
    d2, w2 = bound_daily_drift(0.35, "PT_consensus")
    assert d2 == pytest.approx(0.30)
    assert w2 is not None


def test_computed_prob_up_pct_zero_drift_returns_50() -> None:
    assert computed_prob_up_pct(0.0, 14.0, 5) == pytest.approx(50.0)


def test_computed_prob_up_pct_positive_drift_above_50() -> None:
    p = computed_prob_up_pct(0.10, 14.3, 5)
    assert p > 50.0
    assert p < 60.0


def test_computed_prob_up_pct_grows_then_converges_with_horizon() -> None:
    mu = 0.08
    sig = 12.0
    p0 = computed_prob_up_pct(mu, sig, 0)
    p1 = computed_prob_up_pct(mu, sig, 1)
    p5 = computed_prob_up_pct(mu, sig, 5)
    p20 = computed_prob_up_pct(mu, sig, 20)
    assert p0 == pytest.approx(50.0)
    assert p1 > p0
    assert p5 > p1
    assert p20 > p5
    assert p20 < 100.0


def test_computed_prob_up_pct_zero_sigma_width_returns_50() -> None:
    assert computed_prob_up_pct(1.0, 0.0, 5) == pytest.approx(50.0)


def test_phi_example_nbis_style() -> None:
    """Anchor 179.11 unused; drift 0.10%/day, σ=14.3%, N=5 → Φ(μN/σ)."""
    z = (0.10 * 5) / 14.3
    p = 50.0 * (1.0 + math.erf(z / math.sqrt(2.0)))
    assert p == pytest.approx(51.39, rel=1e-2)
