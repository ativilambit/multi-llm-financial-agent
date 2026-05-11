from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from equity_analyst.config import RunConfig
from equity_analyst.prompt_parts import _load_prompt_file
from equity_analyst.providers.registry import ProviderRegistry
from equity_analyst.types import ProviderResponse

logger = logging.getLogger(__name__)

FACTS_HEADER = "# Market facts (frozen from iteration 1)"

# When extraction fails, still emit the implied-move scaffold so iteration 2+ prompts
# list 1/2/3-sigma forward move rows (values unknown).
_SIGMA = "\u03c3"
_FALLBACK_IMPLIED_MOVES_BLOCK = (
    "- IV / implied moves:\n"
    "  - Post-Earnings IV: unknown\n"
    f"  - Forward 1{_SIGMA} Move: unknown\n"
    f"  - Forward 2{_SIGMA} Move: unknown\n"
    f"  - Forward 3{_SIGMA} Move: unknown\n"
)


def _facts_packet_fallback_markdown(*, reason_bullet: str) -> str:
    return f"{FACTS_HEADER}\n\n{reason_bullet}\n{_FALLBACK_IMPLIED_MOVES_BLOCK}"


def facts_frozen_user_prefix(*, facts_markdown: str) -> str:
    """User-message prefix for fan-out iterations 2+ (discourage duplicate web fetches)."""
    body = facts_markdown.strip()
    return (
        "# FACTS (frozen from iteration 1 — do NOT re-fetch via web_search)\n\n"
        f"{body}\n\n"
        "# TASK\n"
    )


async def extract_facts_packet(*, synthesis_text: str, symbol: str, config: RunConfig) -> str:
    """Call a cheap LLM to distill stable market facts from the latest synthesis markdown."""
    system = _load_prompt_file("facts_extract_system.md")
    user = (
        f"Symbol: {symbol}\n\n"
        f"## Synthesis markdown\n\n{synthesis_text.strip()}\n\n"
        "Respond with markdown only, starting with the required title line.\n"
    )
    prompt = f"{system}\n\n---\n\n{user}"
    reg = ProviderRegistry.default()
    provider = reg.create(
        config.facts_packet_extractor_provider,
        model=config.facts_packet_extractor_model,
        gemini_cache_index=None,
    )
    timeout_s = float(config.request_timeout_s)
    max_out = int(config.facts_packet_max_output_tokens)

    async def _call() -> ProviderResponse:
        return await provider.generate(
            prompt,
            enable_web_search=False,
            max_output_tokens=max_out,
        )

    try:
        resp = await asyncio.wait_for(_call(), timeout=timeout_s)
    except TimeoutError:
        logger.warning("facts_packet: extractor timeout symbol=%s", symbol)
        return _facts_packet_fallback_markdown(
            reason_bullet="- Extraction timed out; treat facts as unknown.",
        )
    except Exception as exc:
        logger.warning("facts_packet: extractor failed symbol=%s err=%r", symbol, exc)
        return _facts_packet_fallback_markdown(
            reason_bullet=f"- Extraction failed ({type(exc).__name__}); treat facts as unknown.",
        )

    text = resp.text.strip()
    if FACTS_HEADER not in text:
        text = f"{FACTS_HEADER}\n\n{text}"
    return text.rstrip() + "\n"


def write_facts_packet(run_dir: Path, markdown: str) -> Path:
    path = run_dir / "facts_packet.md"
    path.write_text(markdown, encoding="utf-8")
    return path
