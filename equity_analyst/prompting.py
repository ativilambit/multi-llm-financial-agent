from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from equity_analyst.config import RunConfig
from equity_analyst.options_chain import (
    apply_options_chain_event_expiry_resolution,
    event_jump_implied_move_pct_from_prompt_dict,
    iv_crush_multiplier,
)
from equity_analyst.outcome_tracker import (
    compute_pead_avg_drift_pct,
    compute_recent_momentum_drift_pct,
    fetch_hv30_annualized_percent,
)
from equity_analyst.prompt_parts import EQUITY_ANALYST_SYSTEM_PROMPT
from equity_analyst.sigma_compute import (
    format_computed_probabilities_reference_markdown,
    resolve_daily_vol_pct_for_sigma,
    try_build_computed_sigma_bundle,
)
from equity_analyst.synthesizer_blend import format_t0_blend_qual_quant_literal

log = logging.getLogger(__name__)


def _resolve_same_day_intraday(cfg: RunConfig) -> tuple[float | None, float | None]:
    """Return (low, high) for the earnings session when config or auto-fetch supplies them."""
    lo, hi = cfg.same_day_intraday_min, cfg.same_day_intraday_max
    if lo is not None and hi is not None:
        return lo, hi
    if not cfg.same_day_intraday_auto_fetch:
        return None, None
    from equity_analyst.outcome_tracker import fetch_earnings_day_intraday_high_low_yfinance

    return fetch_earnings_day_intraday_high_low_yfinance(cfg.symbol, cfg.earnings_date)


def _options_chain_fallback_message(cfg: RunConfig, oc_data: dict[str, Any]) -> str:
    """User-facing sentence when ``options_chain_available`` is false (injected into equity template)."""
    cite = "Use public chain sources (Yahoo Options, CBOE) and label each number with its source."
    if oc_data.get("options_chain_available"):
        return ""
    if not cfg.options_chain_auto_fetch:
        return f"Verified options chain not fetched (auto-fetch disabled); {cite}"
    fe = oc_data.get("fetch_error")
    if isinstance(fe, str) and fe.strip():
        return f"Verified options chain fetch failed ({fe.strip()}); {cite}"
    avail = oc_data.get("available_expiries") or []
    if not avail:
        return (
            "Verified options chain has no listed expiries (likely no options trade on this ticker); "
            f"{cite}"
        )
    return f"Verified options chain unavailable for this run; {cite}"


def _resolve_options_chain(cfg: RunConfig) -> tuple[dict[str, Any], str]:
    """Return ``(options_chain_data dict, options_chain_markdown)`` for Jinja."""
    from equity_analyst.options_chain import (
        empty_options_prompt_dict,
        fetch_options_chain_prompt_dict,
        options_chain_snapshot_from_prompt_dict,
    )

    if cfg.options_chain_snapshot is not None and isinstance(cfg.options_chain_snapshot, dict):
        snap = options_chain_snapshot_from_prompt_dict(cfg.options_chain_snapshot)
        if snap is not None:
            data = snap.to_prompt_dict()
            return data, snap.to_markdown_table(earnings_date=cfg.earnings_date)
        log.warning("options_chain: manual options_chain_snapshot invalid; falling back to fetch/off")

    if cfg.options_chain_auto_fetch:
        data = fetch_options_chain_prompt_dict(
            cfg.symbol,
            cfg.earnings_date,
            cfg.target_dates,
            today_date=cfg.today_date,
            max_weekly_lookahead_days=int(cfg.max_weekly_lookahead_days),
        )
        if data.get("options_chain_available"):
            snap = options_chain_snapshot_from_prompt_dict(data)
            if snap is not None:
                return data, snap.to_markdown_table(earnings_date=cfg.earnings_date)
        return data, ""

    empty = empty_options_prompt_dict(cfg.symbol)
    return empty, ""


def split_static_dynamic(rendered: RenderedPrompt) -> tuple[str, str]:
    """Split equity prompt into (static persona, dynamic user body) matching Anthropic caching."""
    return EQUITY_ANALYST_SYSTEM_PROMPT, rendered.user_message_text


@dataclass(frozen=True)
class RenderedPrompt:
    """Rendered equity template.

    ``text`` is preamble + body for non-Anthropic providers and synthesis.
    ``user_message_text`` is body only (Anthropic caching user turn).
    """

    template_path: str
    text: str
    context: dict[str, Any]
    user_message_text: str


def _derived_context(cfg: RunConfig) -> dict[str, Any]:
    target_dates_joined = ", ".join(cfg.target_dates)
    target_dates_joined_later = ", ".join(cfg.target_dates[1:]) if len(cfg.target_dates) > 1 else ""
    target_dates_first = cfg.target_dates[0] if cfg.target_dates else ""
    target_dates_first_later = cfg.target_dates[1] if len(cfg.target_dates) > 1 else ""
    target_dates_joined_later_first = cfg.target_dates[1] if len(cfg.target_dates) > 1 else ""
    short_interest_lookbacks_joined = ", ".join(cfg.short_interest_lookbacks)

    return {
        "target_dates_joined": target_dates_joined,
        "target_dates_joined_later": target_dates_joined_later,
        "target_dates_first": target_dates_first,
        "target_dates_first_later": target_dates_first_later,
        "target_dates_joined_later_first": target_dates_joined_later_first,
        "short_interest_lookbacks_joined": short_interest_lookbacks_joined,
    }


def render_prompt(cfg: RunConfig, prompt_path: Path, *, prior_synthesis_text: str | None = None) -> RenderedPrompt:
    env = Environment(
        loader=FileSystemLoader(str(prompt_path.parent)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    template = env.get_template(prompt_path.name)

    context: dict[str, Any] = {
        "symbol": cfg.symbol,
        "company_name": cfg.company_name,
        "reference_session_low": cfg.today_low,
        "reference_session_high": cfg.today_high,
        "reference_last_price": cfg.current_price,
        "today_low": cfg.today_low,
        "today_high": cfg.today_high,
        "current_price": cfg.current_price,
        "today_date": cfg.today_date,
        "today_session": cfg.today_session,
        "earnings_date": cfg.earnings_date,
        "earnings_timing": cfg.earnings_timing,
        "target_dates": cfg.target_dates,
        "next_trading_day": cfg.next_trading_day,
        "followup_open_date": cfg.followup_open_date,
        "historical_quarters": cfg.historical_quarters,
        "short_interest_lookbacks": cfg.short_interest_lookbacks,
    }
    context.update(_derived_context(cfg))

    sd_lo, sd_hi = _resolve_same_day_intraday(cfg)
    same_day_intraday_available = sd_lo is not None and sd_hi is not None
    context["same_day_intraday_min"] = sd_lo
    context["same_day_intraday_max"] = sd_hi
    context["same_day_intraday_available"] = same_day_intraday_available
    if sd_lo is not None and sd_hi is not None:
        context["same_day_intraday_anchor_band_low"] = sd_lo - 1.0
        context["same_day_intraday_anchor_band_high"] = sd_hi + 1.0
    else:
        context["same_day_intraday_anchor_band_low"] = None
        context["same_day_intraday_anchor_band_high"] = None

    oc_data, oc_md = _resolve_options_chain(cfg)
    oc_data["max_weekly_lookahead_days"] = int(cfg.max_weekly_lookahead_days)
    apply_options_chain_event_expiry_resolution(
        oc_data,
        earnings_date=cfg.earnings_date,
        max_weekly_lookahead_days=int(cfg.max_weekly_lookahead_days),
    )
    context["options_chain_data"] = oc_data
    context["options_chain_markdown"] = oc_md
    context["options_chain_available"] = bool(oc_data.get("options_chain_available"))
    context["options_chain_fallback_message"] = _options_chain_fallback_message(cfg, oc_data)
    if cfg.options_chain_auto_fetch and not oc_data.get("options_chain_available"):
        log.warning(
            "options_chain: auto_fetch enabled but verified chain unavailable (%s)",
            oc_data.get("fetch_error") or "no fetch_error on payload",
        )

    iv_m = (
        iv_crush_multiplier(oc_data, earnings_date=cfg.earnings_date)
        if oc_data.get("options_chain_available")
        else None
    )
    hv30_pct = fetch_hv30_annualized_percent(cfg.symbol)
    daily_iv_adj: float | None = None
    if iv_m is not None and hv30_pct is not None:
        daily_iv_adj = (hv30_pct / math.sqrt(252.0)) * iv_m
    context["iv_crush_multiplier"] = iv_m
    context["hv30_annualized_pct"] = hv30_pct
    context["daily_vol_iv_adjusted"] = daily_iv_adj

    pead_d = compute_pead_avg_drift_pct(cfg.symbol)
    mom_d = compute_recent_momentum_drift_pct(cfg.symbol, lookback_days=10)
    context["pead_avg_drift_pct"] = pead_d
    context["recent_momentum_drift_pct"] = mom_d

    sig_avail, sig_md, sig_tbl, _sig_tag = try_build_computed_sigma_bundle(
        symbol=cfg.symbol,
        anchor_price=cfg.current_price,
        same_day_intraday_available=same_day_intraday_available,
        earnings_date=cfg.earnings_date,
        earnings_timing=cfg.earnings_timing,
        target_dates=list(cfg.target_dates),
        next_trading_day=cfg.next_trading_day,
        oc_data=oc_data,
        max_weekly_lookahead_days=int(cfg.max_weekly_lookahead_days),
    )
    context["computed_sigma_bands_available"] = sig_avail
    context["computed_sigma_bands_markdown"] = sig_md if sig_avail else ""
    context["computed_sigma_bands_table"] = sig_tbl
    context["options_expiry_class"] = oc_data.get("expiry_class")
    context["options_event_jump_source"] = oc_data.get("event_jump_source")
    context["diffusion_only_sigma_hint"] = bool(
        isinstance(sig_tbl, dict) and sig_tbl.get("diffusion_only_sigma"),
    )

    event_jump_chain_warning_md = ""
    if not sig_avail and oc_data.get("options_chain_available"):
        ej_dbg = event_jump_implied_move_pct_from_prompt_dict(
            oc_data,
            earnings_date=cfg.earnings_date,
            max_weekly_lookahead_days=int(cfg.max_weekly_lookahead_days),
        )
        dv_dbg, _ = resolve_daily_vol_pct_for_sigma(cfg.symbol, oc_data, earnings_date=cfg.earnings_date)
        if ej_dbg is None and dv_dbg is not None:
            event_jump_chain_warning_md = (
                "**Server warning (`event_jump` not computed):** A verified options chain was fetched, but the "
                "server could not derive an **ATM straddle / implied-move percent** for the event-week expiry "
                "(missing or unusable straddle mid; IV fallback also failed). **Do not invent** an `event_jump=` "
                "literal or event-week sigma from narrative — state that the figure is unavailable from the chain "
                "payload or cite specific option rows with URLs. Do **not** substitute a silent pure-HV30√t "
                "envelope when earnings implied volatility is the stated setup."
            )
    context["event_jump_chain_warning_markdown"] = event_jump_chain_warning_md

    prob_md = ""
    if prior_synthesis_text:
        prob_md = format_computed_probabilities_reference_markdown(
            prior_synthesis_text,
            earnings_date=cfg.earnings_date,
            earnings_timing=cfg.earnings_timing,
        )
    context["computed_probabilities_markdown"] = prob_md
    context["t0_blend_literal"] = format_t0_blend_qual_quant_literal(cfg.t0_blend_preset)

    user_message_text = template.render(**context)
    text = f"{EQUITY_ANALYST_SYSTEM_PROMPT}\n\n{user_message_text}"
    return RenderedPrompt(
        template_path=str(prompt_path),
        text=text,
        context=context,
        user_message_text=user_message_text,
    )

