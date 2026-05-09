from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from equity_analyst.config import RunConfig


@dataclass(frozen=True)
class RenderedPrompt:
    template_path: str
    text: str
    context: dict[str, Any]


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

    text = template.render(**context)
    return RenderedPrompt(template_path=str(prompt_path), text=text, context=context)

