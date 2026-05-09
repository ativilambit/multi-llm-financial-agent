from __future__ import annotations

from typing import Any

# Verbatim persona / instruction preamble formerly in prompts/equity_analyst.j2 line 1,
# through the triple-check sentence (dynamic price/session line moved to the user template).
EQUITY_ANALYST_SYSTEM_PROMPT = (
    "as the buy-side top 0.0001% equity investment strategist and analyst and top equity and "
    "options portfolio manager, shows your best equity analysis work by running deep and thoughtful "
    "real-time, checks, analysis, reasoning, and research models, including the best available models "
    "and based on the latest and the most up-to-date time data, step-by-step research and analyses "
    "and today's most recent options price action in . Before answering, triple-check your answers "
    "for accuracy, validity, and correctness to the highest possible level."
)


def ephemeral_cache_control(*, ttl_1h: bool = True) -> dict[str, Any]:
    """Anthropic prompt cache breakpoint; 1h TTL reduces churn for repeated template runs."""
    if ttl_1h:
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}
