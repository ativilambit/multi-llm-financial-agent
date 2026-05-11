from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from equity_analyst.config import RunConfig
from equity_analyst.prompt_parts import _load_prompt_file
from equity_analyst.providers.registry import ProviderRegistry
from equity_analyst.synthesizer import detect_max_tokens_truncation
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

# At least three groups must match (case-insensitive). Alternatives cover prompt wording drift.
_FACTS_SIGNATURE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("Last verified close",),
    ("IV / implied moves",),
    ("Analyst targets",),
    ("Session range", "Session SD targets"),
    ("PCR",),
    ("Short interest",),
    ("Historical Earnings Reactions", "earnings reactions"),
    ("Key Qualitative Anchors", "Key qualitative anchors"),
)

_RETRY_MAX_OUT_CAP = 16_384


def _matched_signature_labels(text: str) -> list[str]:
    hay = text.casefold()
    labels: list[str] = []
    for group in _FACTS_SIGNATURE_GROUPS:
        for alt in group:
            if alt.casefold() in hay:
                labels.append(alt)
                break
    return labels


def _invalid_tail_mid_fragment(s: str) -> bool:
    """Heuristic: calendar tail cut mid-phrase (e.g. 'Tue May' with no day/year)."""
    core = s.rstrip()
    if not core:
        return False
    last_line = core.split("\n")[-1].strip()
    return bool(
        re.search(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+[A-Z][a-z]{0,3}\s*$", last_line)
    )


def _first_body_line_looks_truncated(s: str) -> bool:
    """True when the first non-empty line after the title looks like a mid-sentence fragment."""
    body = s.strip()
    if not body.startswith(FACTS_HEADER):
        return True
    rest = body.removeprefix(FACTS_HEADER).lstrip("\n")
    first = ""
    for ln in rest.split("\n"):
        if ln.strip():
            first = ln.strip()
            break
    if not first:
        return True
    return first[0] in ")±"


def _facts_packet_markdown_valid(text: str) -> tuple[bool, list[str]]:
    """Return (ok, sections_found) using structure / truncation heuristics."""
    s = text.strip()
    sections = _matched_signature_labels(s)
    if not s.startswith(FACTS_HEADER):
        return False, sections
    if not text.rstrip(" \t").endswith("\n"):
        return False, sections
    if len(sections) < 3:
        return False, sections
    if _first_body_line_looks_truncated(s):
        return False, sections
    if _invalid_tail_mid_fragment(s):
        return False, sections
    return True, sections


def _heuristic_truncation_warrants_retry(text: str, sections: list[str]) -> bool:
    """True when output looks cut off even without an explicit MAX_TOKENS signal."""
    s = text.strip()
    if _invalid_tail_mid_fragment(s):
        return True
    if _first_body_line_looks_truncated(s):
        return True
    if len(sections) < 3:
        return True
    return not text.rstrip(" \t").endswith("\n") and len(s) > 80


def _normalize_facts_markdown(text: str) -> str:
    s = text.strip()
    return s.rstrip("\n") + "\n"


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

    async def _call(max_tokens: int) -> ProviderResponse:
        return await provider.generate(
            prompt,
            enable_web_search=False,
            max_output_tokens=max_tokens,
        )

    try:
        resp = await asyncio.wait_for(_call(max_out), timeout=timeout_s)
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

    def _emit_truncation_warning(t: str, sections: list[str]) -> None:
        out_chars = len(t.strip())
        logger.warning(
            "facts_packet: output_chars=%s looks truncated/malformed sections_found=%s",
            out_chars,
            sections,
        )

    async def _finalize_or_retry(first: ProviderResponse) -> str:
        t = first.text
        ok, sections = _facts_packet_markdown_valid(t)
        if ok:
            out = _normalize_facts_markdown(t)
            logger.info(
                "Facts packet extracted chars=%s sections_found=%s",
                len(out.strip()),
                sections,
            )
            return out

        _emit_truncation_warning(t, sections)
        max_hit, fr_label = detect_max_tokens_truncation(first.raw)
        should_retry = max_hit or _heuristic_truncation_warrants_retry(t, sections)
        retry_max = min(max_out * 2, _RETRY_MAX_OUT_CAP)
        if not should_retry or retry_max <= max_out:
            return _facts_packet_fallback_markdown(
                reason_bullet="- Facts packet extraction returned malformed output; treat facts as unknown.",
            )

        logger.info(
            "facts_packet: retrying extraction symbol=%s max_output_tokens=%s (was=%s finish_reason=%s)",
            symbol,
            retry_max,
            max_out,
            fr_label,
        )
        try:
            second = await asyncio.wait_for(_call(retry_max), timeout=timeout_s)
        except TimeoutError:
            logger.warning("facts_packet: extractor retry timeout symbol=%s", symbol)
            return _facts_packet_fallback_markdown(
                reason_bullet="- Extraction retry timed out; treat facts as unknown.",
            )
        except Exception as exc:
            logger.warning("facts_packet: extractor retry failed symbol=%s err=%r", symbol, exc)
            return _facts_packet_fallback_markdown(
                reason_bullet=f"- Extraction retry failed ({type(exc).__name__}); treat facts as unknown.",
            )

        t2 = second.text
        ok2, sections2 = _facts_packet_markdown_valid(t2)
        if ok2:
            out = _normalize_facts_markdown(t2)
            logger.info(
                "Facts packet extracted chars=%s sections_found=%s",
                len(out.strip()),
                sections2,
            )
            return out

        _emit_truncation_warning(t2, sections2)
        return _facts_packet_fallback_markdown(
            reason_bullet="- Facts packet extraction remained malformed after retry; treat facts as unknown.",
        )

    return await _finalize_or_retry(resp)


def write_facts_packet(run_dir: Path, markdown: str) -> Path:
    path = run_dir / "facts_packet.md"
    path.write_text(markdown, encoding="utf-8")
    return path
