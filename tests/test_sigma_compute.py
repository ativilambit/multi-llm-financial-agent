from __future__ import annotations

import json

import pytest

from equity_analyst.sigma_compute import (
    compute_sigma_bands_server_side,
    format_computed_sigma_bands_markdown,
    resolve_daily_vol_pct_for_sigma,
    try_build_computed_sigma_bundle,
    verify_emitted_sigma_bands_match_computed,
)


def test_compute_sigma_bands_server_side_variance_additive() -> None:
    t = compute_sigma_bands_server_side(
        anchor_price=100.0,
        anchor_type="prior_close",
        earnings_date="Wed May 13 2026",
        earnings_timing=None,
        target_dates=["2026-05-13", "2026-05-19"],
        next_trading_day="2026-05-14",
        event_jump_pct=6.0,
        daily_vol_pct=8.0,
        daily_vol_source="test",
        daily_drift_pct=0.0,
        drift_source_note="test",
    )
    assert t is not None
    by = {s.session_date.isoformat(): s for s in t.sessions}
    s_may19 = by["2026-05-19"]
    expected = (36.0 + 5.0 * 8.0**2) ** 0.5
    assert s_may19.one_sigma_half_width_pct == pytest.approx(expected, rel=1e-6)


def test_compute_sigma_bands_with_iv_crush_multiplier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "equity_analyst.sigma_compute.fetch_hv30_annualized_percent",
        lambda _s: 84.9,
    )
    monkeypatch.setattr("equity_analyst.sigma_compute.iv_crush_multiplier", lambda *_a, **_k: 0.59)
    dv, src = resolve_daily_vol_pct_for_sigma(
        "NBIS",
        {"options_chain_available": True, "selected_expiries": []},
        earnings_date="2026-05-13",
    )
    assert src == "HV30/sqrt252*iv_crush_multiplier"
    raw = 84.9 / (252**0.5)
    assert dv == pytest.approx(raw * 0.59, rel=1e-6)


def test_computed_sigma_bands_table_injected_when_chain_and_hv30_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "equity_analyst.sigma_compute.fetch_hv30_annualized_percent",
        lambda _s: 50.0,
    )
    monkeypatch.setattr(
        "equity_analyst.sigma_compute.event_jump_implied_move_pct_from_prompt_dict",
        lambda *_a, **_k: 10.0,
    )
    monkeypatch.setattr(
        "equity_analyst.sigma_compute.compute_pead_avg_drift_pct",
        lambda _s: 0.08,
    )
    monkeypatch.setattr(
        "equity_analyst.sigma_compute.compute_recent_momentum_drift_pct",
        lambda *_a, **_k: None,
    )
    oc = {
        "options_chain_available": True,
        "spot": 100.0,
        "selected_expiries": [
            {
                "expiry_date": "2026-05-15",
                "implied_move_pct": 10.0,
                "atm_straddle_mid": 10.0,
            },
        ],
    }
    ok, md, tbl, _tag = try_build_computed_sigma_bundle(
        symbol="X",
        anchor_price=100.0,
        same_day_intraday_available=False,
        earnings_date="2026-05-13",
        earnings_timing=None,
        target_dates=["2026-05-13"],
        next_trading_day="2026-05-14",
        oc_data=oc,
    )
    assert ok is True
    assert "Server-computed" in md
    assert isinstance(tbl, dict)
    assert tbl["sessions"]


def test_computed_sigma_bands_falls_back_when_missing_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "equity_analyst.sigma_compute.fetch_hv30_annualized_percent",
        lambda _s: None,
    )
    monkeypatch.setattr(
        "equity_analyst.sigma_compute.compute_realized_post_earnings_daily_vol_pct",
        lambda _s: None,
    )
    monkeypatch.setattr(
        "equity_analyst.sigma_compute.event_jump_implied_move_pct_from_prompt_dict",
        lambda *_a, **_k: None,
    )
    ok, md, tbl, _tag = try_build_computed_sigma_bundle(
        symbol="X",
        anchor_price=100.0,
        same_day_intraday_available=False,
        earnings_date="2026-05-13",
        earnings_timing=None,
        target_dates=["2026-05-13"],
        next_trading_day="2026-05-14",
        oc_data={"options_chain_available": False},
    )
    assert ok is False
    assert md == ""
    assert tbl is None


def test_verify_emitted_sigma_bands_strict_equality_within_tolerance() -> None:
    comp = {
        "sessions": [
            {
                "session_date": "2026-05-13",
                "one_sigma_half_width_pct": 11.31,
                "three_sigma_half_width_pct": 33.93,
            },
        ],
    }
    syn = (
        "```json\n"
        + json.dumps(
            {
                "sigma_summary": {
                    "anchor_price": 179.11,
                    "anchor_type": "prior_close",
                    "sessions": [
                        {
                            "date": "2026-05-13",
                            "label": "T0",
                            "N": 0,
                            "one_sigma_half_width_pct": 11.31,
                            "three_sigma_half_width_pct": 33.93,
                        },
                    ],
                },
            },
        )
        + "\n```\n"
    )
    assert verify_emitted_sigma_bands_match_computed(syn, comp, tolerance_pp=1.0) == []


def test_verify_emitted_sigma_bands_flags_drift_beyond_1pp() -> None:
    comp = {
        "sessions": [
            {
                "session_date": "2026-05-13",
                "one_sigma_half_width_pct": 11.31,
                "three_sigma_half_width_pct": 33.93,
            },
        ],
    }
    syn = (
        "```json\n"
        + json.dumps(
            {
                "sigma_summary": {
                    "anchor_price": 179.11,
                    "anchor_type": "prior_close",
                    "sessions": [
                        {
                            "date": "2026-05-13",
                            "label": "T0",
                            "N": 0,
                            "one_sigma_half_width_pct": 20.0,
                            "three_sigma_half_width_pct": 60.0,
                        },
                    ],
                },
            },
        )
        + "\n```\n"
    )
    qs = verify_emitted_sigma_bands_match_computed(syn, comp, tolerance_pp=1.0)
    assert len(qs) >= 1
    assert "1σ half-width" in qs[0]


def test_format_computed_sigma_bands_markdown_contains_rows() -> None:
    t = compute_sigma_bands_server_side(
        anchor_price=179.11,
        anchor_type="prior_close",
        earnings_date="2026-05-13",
        earnings_timing=None,
        target_dates=["2026-05-13", "2026-05-14", "2026-05-15", "2026-05-20"],
        next_trading_day="2026-05-14",
        event_jump_pct=11.31,
        daily_vol_pct=4.54,
        daily_vol_source="fixture",
        daily_drift_pct=0.10,
        drift_source_note="fixture",
    )
    assert t is not None
    md = format_computed_sigma_bands_markdown(t)
    assert "2026-05-13" in md
    assert "11.31" in md or "4.54" in md
