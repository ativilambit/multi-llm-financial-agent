from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from equity_analyst.config import RunConfig
from equity_analyst.prompt_export import logical_prompt_split, prompt_call_context
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

_EXPECTED_SECTION_GROUPS = len(_FACTS_SIGNATURE_GROUPS)
_PASS_A_MIN_SECTIONS = 6
_PASS_A_MIN_CHARS = 1500
_FAIL_CHARS = 800
_FAIL_MAX_SECTIONS = 2
_FALLBACK_MAX_CHARS = 500


class FactsPacketDecision(StrEnum):
    ACCEPT = "ACCEPT"
    ACCEPT_WITH_REVIEW = "ACCEPT_WITH_REVIEW"
    RETRY = "RETRY"
    FALLBACK = "FALLBACK"


@dataclass(frozen=True)
class FactsPacketEval:
    """Structured result from facts-packet validation heuristics."""

    decision: FactsPacketDecision
    sections_found: tuple[str, ...]
    output_chars: int
    heuristic_flags: tuple[str, ...]
    pass_gate_a: bool
    pass_gate_b: bool


def _matched_signature_labels(text: str) -> list[str]:
    hay = text.casefold()
    labels: list[str] = []
    for group in _FACTS_SIGNATURE_GROUPS:
        for alt in group:
            if alt.casefold() in hay:
                labels.append(alt)
                break
    return labels


def _invalid_tail_weekday_fragment_near_eof(s: str, *, window: int = 50) -> bool:
    """Calendar phrase cut mid-date — only inspect the last `window` chars (true tail cut)."""
    core = s.rstrip()
    if not core:
        return False
    tail = core[-window:] if len(core) >= window else core
    last_line = tail.split("\n")[-1].strip()
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


def _explicit_complete_end(text: str) -> bool:
    """Gate (b): output reads explicitly complete at the tail."""
    core = text.rstrip()
    if not core:
        return False
    last = core[-1]
    return last in ".!?)>" or last == "|"


def _clear_truncation_markers(text: str) -> list[str]:
    """Strong truncation signals (contribute to strict structural fail when gates A/B are false)."""
    flags: list[str] = []
    core = text.rstrip()
    if not core:
        return ["empty"]
    if core.endswith("..."):
        flags.append("ends_with_ellipsis")
    tail = core[-120:] if len(core) >= 120 else core
    if tail.count("(") > tail.count(")"):
        flags.append("unbalanced_open_paren_near_tail")
    if re.search(r"\$\d+\.\s*$", core):
        flags.append("money_dot_without_following_digits")
    return flags


def _tail_open_paren_unbalanced(text: str, *, window: int = 50) -> bool:
    """Unclosed '(' in the last window — retry signal for otherwise rich packets."""
    core = text.rstrip()
    if not core:
        return False
    tail = core[-window:] if len(core) >= window else core
    return tail.count("(") > tail.count(")")


def _soft_heuristic_flags(stripped: str, raw_text: str) -> list[str]:
    """Signals for logging / ACCEPT_WITH_REVIEW (not used to reject gate A)."""
    flags: list[str] = []
    if _first_body_line_looks_truncated(stripped):
        flags.append("first_body_line_fragment")
    if not raw_text.rstrip(" \t").endswith("\n"):
        flags.append("missing_trailing_newline")
    if _invalid_tail_weekday_fragment_near_eof(stripped):
        flags.append("tail_weekday_fragment")
    if _tail_open_paren_unbalanced(raw_text):
        flags.append("tail_unbalanced_open_paren")
    return sorted(set(flags))


def _strict_structural_fail(text: str, sections: list[str]) -> bool:
    """
    FAIL (strict) if ANY of:
    - chars < 800
    - 0-2 expected sections
    - clear truncation markers
    """
    chars = len(text.strip())
    sec_n = len(sections)
    if chars < _FAIL_CHARS:
        return True
    if sec_n <= _FAIL_MAX_SECTIONS:
        return True
    return bool(_clear_truncation_markers(text))


def _pass_gate_a(text: str, sections: list[str]) -> bool:
    return len(sections) >= _PASS_A_MIN_SECTIONS and len(text.strip()) >= _PASS_A_MIN_CHARS


def _pass_gate_b(text: str) -> bool:
    return _explicit_complete_end(text)


def _merged_flags(soft: list[str], trunc: list[str]) -> tuple[str, ...]:
    return tuple(sorted(set(soft) | set(trunc)))


def evaluate_facts_packet(text: str) -> FactsPacketEval:
    """
    Acceptance:
    - (a) ≥6 of 8 sections and ≥1500 chars → ACCEPT (keeps legitimate ':' / header tails).
    - Else (b) explicit closing tail punctuation / table bar → ACCEPT or ACCEPT_WITH_REVIEW if soft flags.
    - Else strict structural fail → RETRY.
    - Else ambiguous but not hopeless → ACCEPT_WITH_REVIEW.
    """
    s = text.strip()
    sections = _matched_signature_labels(s)
    chars = len(s)
    gate_a = _pass_gate_a(text, sections)
    gate_b = _pass_gate_b(text)
    soft = _soft_heuristic_flags(s, text)
    trunc = _clear_truncation_markers(text)
    flags = _merged_flags(soft, trunc)

    if gate_a:
        return FactsPacketEval(
            decision=FactsPacketDecision.ACCEPT,
            sections_found=tuple(sections),
            output_chars=chars,
            heuristic_flags=flags,
            pass_gate_a=True,
            pass_gate_b=gate_b,
        )

    if gate_b:
        decision = FactsPacketDecision.ACCEPT_WITH_REVIEW if soft or trunc else FactsPacketDecision.ACCEPT
        return FactsPacketEval(
            decision=decision,
            sections_found=tuple(sections),
            output_chars=chars,
            heuristic_flags=flags,
            pass_gate_a=False,
            pass_gate_b=True,
        )

    if _strict_structural_fail(text, sections):
        return FactsPacketEval(
            decision=FactsPacketDecision.RETRY,
            sections_found=tuple(sections),
            output_chars=chars,
            heuristic_flags=flags,
            pass_gate_a=False,
            pass_gate_b=False,
        )

    # Rich packet that failed gates A/B but looks mechanically cut off — worth one doubled-budget retry.
    if (
        len(sections) >= _PASS_A_MIN_SECTIONS
        and chars >= _FAIL_CHARS
        and _heuristic_truncation_warrants_retry(text, sections)
    ):
        return FactsPacketEval(
            decision=FactsPacketDecision.RETRY,
            sections_found=tuple(sections),
            output_chars=chars,
            heuristic_flags=flags,
            pass_gate_a=False,
            pass_gate_b=False,
        )

    return FactsPacketEval(
        decision=FactsPacketDecision.ACCEPT_WITH_REVIEW,
        sections_found=tuple(sections),
        output_chars=chars,
        heuristic_flags=tuple([*flags, "ambiguous_tail_no_gate_ab"]),
        pass_gate_a=False,
        pass_gate_b=False,
    )


def _genuinely_unusable_for_fallback(text: str, sections: list[str]) -> bool:
    """Emit unknown template only for empty / near-empty extractions."""
    chars = len(text.strip())
    sec_n = len(sections)
    return (chars < _FALLBACK_MAX_CHARS and sec_n == 0) or (
        sec_n <= _FAIL_MAX_SECTIONS and chars < _FAIL_CHARS
    )


def _heuristic_truncation_warrants_retry(text: str, sections: list[str]) -> bool:
    """True when output looks cut off enough to try a larger budget (without MAX_TOKENS)."""
    s = text.strip()
    if _invalid_tail_weekday_fragment_near_eof(s):
        return True
    if _first_body_line_looks_truncated(s):
        return True
    if len(sections) < 3:
        return True
    if _tail_open_paren_unbalanced(text) and len(sections) >= _PASS_A_MIN_SECTIONS:
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


def _log_facts_decision(
    *,
    symbol: str,
    eval_result: FactsPacketEval,
    max_tokens_hit: bool,
    finish_reason_label: str | None,
) -> None:
    flags = list(eval_result.heuristic_flags)
    parts = [
        f"output_chars={eval_result.output_chars}",
        f"sections={len(eval_result.sections_found)}/{_EXPECTED_SECTION_GROUPS}",
        f"gate_a={eval_result.pass_gate_a}",
        f"gate_b={eval_result.pass_gate_b}",
        f"max_tokens_hit={max_tokens_hit}",
        f"finish_reason={finish_reason_label or 'n/a'}",
        f"heuristic_flags={flags}",
        f"decision={eval_result.decision}",
    ]
    msg = "facts_packet: " + " ".join(parts)
    if eval_result.decision in (FactsPacketDecision.ACCEPT_WITH_REVIEW, FactsPacketDecision.RETRY):
        logger.warning("%s symbol=%s", msg, symbol)
    else:
        logger.info("%s symbol=%s", msg, symbol)


def _finalize_markdown_or_fallback(
    *,
    symbol: str,
    text: str,
    eval_result: FactsPacketEval,
    after_retry: bool,
) -> str:
    """Prefer keeping partial-but-useful output over the unknown template."""
    if eval_result.decision in (FactsPacketDecision.ACCEPT, FactsPacketDecision.ACCEPT_WITH_REVIEW):
        return _normalize_facts_markdown(text)
    if not _genuinely_unusable_for_fallback(text, list(eval_result.sections_found)):
        kept = _normalize_facts_markdown(text)
        logger.warning(
            "facts_packet: keeping_extractor_output_despite_flags symbol=%s after_retry=%s chars=%s sections=%s",
            symbol,
            after_retry,
            eval_result.output_chars,
            len(eval_result.sections_found),
        )
        return kept
    reason = (
        "- Facts packet extraction remained malformed after retry; treat facts as unknown."
        if after_retry
        else "- Facts packet extraction returned malformed output; treat facts as unknown."
    )
    return _facts_packet_fallback_markdown(reason_bullet=reason)


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
        with prompt_call_context(node="facts_extract", iteration=1), logical_prompt_split(system, user):
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

    async def _finalize_or_retry(first: ProviderResponse) -> str:
        t = first.text
        if not t.strip().startswith(FACTS_HEADER):
            logger.warning(
                "facts_packet: missing_required_title decision=%s symbol=%s",
                FactsPacketDecision.FALLBACK,
                symbol,
            )
            return _facts_packet_fallback_markdown(
                reason_bullet="- Facts packet extraction returned malformed output; treat facts as unknown.",
            )

        max_hit, fr_label = detect_max_tokens_truncation(first.raw)
        ev = evaluate_facts_packet(t)
        _log_facts_decision(
            symbol=symbol,
            eval_result=ev,
            max_tokens_hit=max_hit,
            finish_reason_label=fr_label,
        )

        if ev.decision in (FactsPacketDecision.ACCEPT, FactsPacketDecision.ACCEPT_WITH_REVIEW):
            out = _normalize_facts_markdown(t)
            logger.info(
                "Facts packet extracted chars=%s sections_found=%s",
                len(out.strip()),
                list(ev.sections_found),
            )
            return out

        should_retry = max_hit or _heuristic_truncation_warrants_retry(t, list(ev.sections_found))
        retry_max = min(max_out * 2, _RETRY_MAX_OUT_CAP)
        if not should_retry or retry_max <= max_out:
            fb = _finalize_markdown_or_fallback(symbol=symbol, text=t, eval_result=ev, after_retry=False)
            if "treat facts as unknown" in fb:
                logger.warning("facts_packet: decision=%s symbol=%s", FactsPacketDecision.FALLBACK, symbol)
            return fb

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
        if not t2.strip().startswith(FACTS_HEADER):
            logger.warning(
                "facts_packet: missing_required_title decision=%s symbol=%s",
                FactsPacketDecision.FALLBACK,
                symbol,
            )
            return _facts_packet_fallback_markdown(
                reason_bullet="- Facts packet extraction remained malformed after retry; treat facts as unknown.",
            )

        max_hit2, fr_label2 = detect_max_tokens_truncation(second.raw)
        ev2 = evaluate_facts_packet(t2)
        _log_facts_decision(
            symbol=symbol,
            eval_result=ev2,
            max_tokens_hit=max_hit2,
            finish_reason_label=fr_label2,
        )

        if ev2.decision in (FactsPacketDecision.ACCEPT, FactsPacketDecision.ACCEPT_WITH_REVIEW):
            out = _normalize_facts_markdown(t2)
            logger.info(
                "Facts packet extracted chars=%s sections_found=%s",
                len(out.strip()),
                list(ev2.sections_found),
            )
            return out

        fb2 = _finalize_markdown_or_fallback(symbol=symbol, text=t2, eval_result=ev2, after_retry=True)
        if "treat facts as unknown" in fb2:
            logger.warning("facts_packet: decision=%s symbol=%s", FactsPacketDecision.FALLBACK, symbol)
        return fb2

    return await _finalize_or_retry(resp)


def write_facts_packet(run_dir: Path, markdown: str) -> Path:
    path = run_dir / "facts_packet.md"
    path.write_text(markdown, encoding="utf-8")
    return path
