from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from equity_analyst.config import RunConfig
from equity_analyst.prompt_parts import EQUITY_ANALYST_SYSTEM_PROMPT

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
            return data, snap.to_markdown_table()
        log.warning("options_chain: manual options_chain_snapshot invalid; falling back to fetch/off")

    if cfg.options_chain_auto_fetch:
        data = fetch_options_chain_prompt_dict(
            cfg.symbol,
            cfg.earnings_date,
            cfg.target_dates,
            today_date=cfg.today_date,
        )
        if data.get("options_chain_available"):
            snap = options_chain_snapshot_from_prompt_dict(data)
            if snap is not None:
                return data, snap.to_markdown_table()
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


def render_prompt(cfg: RunConfig, prompt_path: Path) -> RenderedPrompt:
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
    context["options_chain_data"] = oc_data
    context["options_chain_markdown"] = oc_md
    context["options_chain_available"] = bool(oc_data.get("options_chain_available"))
    context["options_chain_fallback_message"] = _options_chain_fallback_message(cfg, oc_data)
    if cfg.options_chain_auto_fetch and not oc_data.get("options_chain_available"):
        log.warning(
            "options_chain: auto_fetch enabled but verified chain unavailable (%s)",
            oc_data.get("fetch_error") or "no fetch_error on payload",
        )

    user_message_text = template.render(**context)
    text = f"{EQUITY_ANALYST_SYSTEM_PROMPT}\n\n{user_message_text}"
    return RenderedPrompt(
        template_path=str(prompt_path),
        text=text,
        context=context,
        user_message_text=user_message_text,
    )

